from typing import Annotated, get_args, get_origin, get_type_hints

from mashumaro.config import BaseConfig
from mashumaro.exceptions import InvalidFieldValue
from mashumaro.mixins.dict import DataClassDictMixin
from mashumaro.mixins.json import DataClassJSONMixin
from mashumaro.mixins.toml import DataClassTOMLMixin
from mashumaro.mixins.yaml import DataClassYAMLMixin


class ConfigObject(
    DataClassJSONMixin, DataClassYAMLMixin, DataClassTOMLMixin, DataClassDictMixin
):
    """Base mixin that adds JSON, YAML, TOML, and dict serialisation.

    Subclasses inherit ``to_json``, ``to_yaml``, ``to_toml``, and
    ``to_dict`` methods from mashumaro. Serialisation drops ``None`` fields
    and uses field aliases where defined.
    """

    def __post_init__(self) -> None:

        # Extract validation hints from the class definition. Things like
        # Annotated[float, is_positive]: the hint parser pulls out is_positive
        # and then runs it over the values.

        hints = get_type_hints(self.__class__, include_extras=True)

        for field_name, hint in hints.items():
            if get_origin(hint) is not Annotated:
                continue

            args = get_args(hint)
            type = args[0]
            metadata_args = args[1:]

            for validator in metadata_args:
                if not callable(validator):
                    continue

                current_value = getattr(self, field_name)

                try:
                    res = validator(current_value)
                    if res is not None:
                        setattr(self, field_name, res)
                except (ValueError, TypeError) as e:
                    raise InvalidFieldValue(
                        field_name=field_name,
                        field_type=type,
                        field_value=current_value,
                        holder_class=self.__class__,
                        msg=str(e),
                    ) from e

    class Config(BaseConfig):
        serialize_by_alias = True
        omit_none = True
        forbid_extra_keys = True
