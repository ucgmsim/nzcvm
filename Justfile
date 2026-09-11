set dotenv-load

# Default resolution
res := "500"
smoothing := "10000.0"
construct := "uv run nzcvm basin main"
coastline := "resources/coastline.wkb.gz"

banks:
    if [ ! -d models/BanksPeninsula_GTL.zarr ]; then \
        {{ construct }} ${NZCVM_DATA_ROOT}/regional/BanksPeninsulaVolcanics/BanksPeninsulaVolcanics_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/regional/Canterbury/CantDEM.h5 ${NZCVM_DATA_ROOT}/regional/BanksPeninsulaVolcanics/BanksPeninsulaVolcanics_basement_WGS84.h5 ${NZCVM_DATA_ROOT}/regional/BanksPeninsulaVolcanics/BanksPeninsulaVolcanics_Miocene_WGS84.h5 models/BanksPeninsula.zarr --priority 44 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/banks_dummy.fd_modfile --no-pad-top; \
        uv run nzcvm/scripts/banks.py; \
        rm -r models/BanksPeninsula.zarr; \
    fi

canterbury:
    @test -d models/CantQuatenary.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/regional/Canterbury/CantDEM.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/CantDEM.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Pliocene_46_WGS84_v8p9p18.h5 models/CantQuatenary.zarr --priority 46 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v3_Pliocene_Enforced.fd_modfile  --smoothing {{ smoothing }} --coastline {{ coastline }} 
    @test -d models/CantPliocene.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/regional/Canterbury/CantDEM.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Pliocene_46_WGS84_v8p9p18.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Miocene_WGS84.h5 models/CantPliocene.zarr --priority 47 --rho 1950 --vp 2100 --vs 677 --qs 33.85 --qp 67.7 --smoothing {{ smoothing }} --coastline {{ coastline }} --no-pad-top
    @test -d models/CantMiocene.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/regional/Canterbury/CantDEM.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Miocene_WGS84.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Paleogene_WGS84.h5 models/CantMiocene.zarr --rho 2090 --vp 2500 --vs 984 --qs 49.2 --qp 98.4 --priority 48 --smoothing {{ smoothing }} --coastline {{ coastline }} --no-pad-top
    @test -d models/CantPaleogene.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/regional/Canterbury/CantDEM.h5  ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Paleogene_WGS84.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_basement_WGS84.h5 models/CantPaleogene.zarr --rho 2190 --vp 2850 --vs 1281 --qs 64.05 --qp 128.1 --priority 49 --smoothing {{ smoothing }} --coastline {{ coastline }} --no-pad-top

hanmer:
    @test -d models/Hanmer_Basin.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Hanmer/Hanmer_outline_WGS84_v25p3.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Hanmer/Hanmer_basement_WGS84_v25p3.h5 models/Hanmer_Basin.zarr --priority 41 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

mackenzie:
    @test -d models/Mackenzie.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Mackenzie/Mackenzie_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Mackenzie/Mackenzie_basement_WGS84.h5 models/Mackenzie.zarr --priority 35 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

southland:
    @test -d models/Southland_Basin_1.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Southland/Southland_outline_WGS84_1.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Southland/Southland_basement_WGS84.h5 models/Southland_Basin_1.zarr --priority 10 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 
    @test -d models/Southland_Basin_2.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Southland/Southland_outline_WGS84_2.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Southland/Southland_basement_WGS84.h5 models/Southland_Basin_2.zarr --priority 10 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

west_coast:
    @test -d models/WestCoast.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/WestCoast/WestCoast_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/WestCoast/WestCoast_basement_WGS84.h5 models/WestCoast.zarr --priority 6 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile --smoothing {{ smoothing }} --coastline {{ coastline }}  

te_anau:
    @test -d models/TeAnau.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/TeAnau/TeAnau_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/TeAnau/TeAnau_basement_WGS84.h5 models/TeAnau.zarr --priority 9 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

balclutha:
    @test -d models/Balclutha.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Balclutha/Balclutha_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Balclutha/Balclutha_basement_WGS84.h5 models/Balclutha.zarr --priority 29 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

castle_hill:
    @test -d models/CastleHill.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/CastleHill/CastleHill_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/CastleHill/CastleHill_basement_WGS84.h5 models/CastleHill.zarr --priority 3 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile

cheviot:
    @test -d models/Cheviot.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Cheviot/Cheviot_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Cheviot/Cheviot_basement_WGS84.h5 models/Cheviot.zarr --priority 42 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

collingwood:
    @for i in 1 2 3; do \
        test -d models/Collingwood${i}.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Collingwood/Collingwood_outline_WGS84_${i}.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Collingwood/Collingwood_basement_WGS84.h5 models/Collingwood${i}.zarr --priority 23 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile ; \
    done

hawkes_bay:
    @for i in 1 2 3 4; do \
        test -d models/HawkesBay${i}.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/HawkesBay/HawkesBay_outline_WGS84_${i}.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/HawkesBay/HawkesBay_basement_WGS84.h5 models/HawkesBay${i}.zarr --priority 21 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile  --smoothing {{ smoothing }} --coastline {{ coastline }}; \
    done

southern_hawkes_bay:
    @test -d models/SouthernHawkesBay.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/SouthernHawkesBay/SouthernHawkesBay_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/SouthernHawkesBay/SouthernHawkesBay_basement_WGS84.h5 models/SouthernHawkesBay.zarr --priority 16 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

te_araroa:
    @test -d models/TeAraroa.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/TeAraroa/TeAraroa_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/TeAraroa/TeAraroa_basement_WGS84.h5 models/TeAraroa.zarr --priority 1 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

whangaparoa:
    @test -d models/Whangaparoa.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Whangaparoa/Whangaparoa_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Whangaparoa/Whangaparoa_basement_WGS84.h5 models/Whangaparoa.zarr --priority 13 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

whakatane:
    @test -d models/Whakatane.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Whakatane/Whakatane_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Whakatane/Whakatane_basement_WGS84_v25p8.h5 models/Whakatane.zarr --priority 2 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

napier:
    @for i in 1 2 3 4 5 6; do \
        test -d models/Napier${i}.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Napier/Napier_outline_WGS84_${i}.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Napier/Napier_basement_WGS84.h5 models/Napier${i}.zarr --priority 20 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile  --smoothing {{ smoothing }} --coastline {{ coastline }}; \
    done

porirua:
    @for i in 1 2; do \
        test -d models/Porirua${i}.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Porirua/Porirua_outline_WGS84_${i}.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Porirua/Porirua_basement_WGS84.h5 models/Porirua${i}.zarr --priority 18 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile  --smoothing {{ smoothing }} --coastline {{ coastline }}; \
    done

queen_charlotte:
    @test -d models/QueenCharlotte.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/QueenCharlotte/QueenCharlotte_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/QueenCharlotte/QueenCharlotte_basement_WGS84_v25p8.h5 models/QueenCharlotte.zarr --priority 4 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

greater_wellington:
    @for i in 1 2 3 4 5 6; do \
        test -d models/GreaterWellington${i}.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/GreaterWellington/GreaterWellington_outline_WGS84_${i}.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/GreaterWellington/GreaterWellington_basement_WGS84.h5 models/GreaterWellington${i}.zarr --priority 19 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile  --smoothing {{ smoothing }} --coastline {{ coastline }}; \
    done

ne_otago:
    @for i in 1 2 3 4 5; do \
        test -d models/NE_Otago${i}.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/NE_Otago/NE_Otago_outline_WGS84_${i}.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/NE_Otago/NE_Otago_basement_WGS84.h5 models/NE_Otago${i}.zarr --priority 31 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile ; \
    done

dunedin:
    @test -d models/Dunedin.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Dunedin/Dunedin_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Dunedin/Dunedin_basement_WGS84.h5 models/Dunedin.zarr --priority 28 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile  --smoothing {{ smoothing }} --coastline {{ coastline }}

alexandra:
    @test -d models/Alexandra.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Alexandra/Alexandra_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Alexandra/Alexandra_basement_WGS84.h5 models/Alexandra.zarr --priority 33 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

gisborne:
    @test -d models/Gisborne.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Gisborne/Gisborne_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Gisborne/Gisborne_basement_WGS84.h5 models/Gisborne.zarr --priority 17 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

hakataramea:
    @test -d models/Hakataramea.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Hakataramea/Hakataramea_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Hakataramea/Hakataramea_basement_WGS84.h5 models/Hakataramea.zarr --priority 25 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

karamea:
    @test -d models/Karamea.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Karamea/Karamea_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Karamea/Karamea_basement_WGS84.h5 models/Karamea.zarr --priority 24 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

marlborough:
    @test -d models/Marlborough.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Marlborough/Marlborough_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Marlborough/Marlborough_basement_WGS84.h5 models/Marlborough.zarr --priority 40 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

mosgiel:
    @test -d models/Mosgiel.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Mosgiel/Mosgiel_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Mosgiel/Mosgiel_basement_WGS84.h5 models/Mosgiel.zarr --priority 30 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

murchison:
    @test -d models/Murchison.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Murchison/Murchison_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Murchison/Murchison_basement_WGS84.h5 models/Murchison.zarr --priority 27 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

ranfurly:
    @test -d models/Ranfurly.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Ranfurly/Ranfurly_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Ranfurly/Ranfurly_basement_WGS84.h5 models/Ranfurly.zarr --priority 32 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

rarakau:
    @test -d models/Rarakau.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Rarakau/Rarakau_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Rarakau/Rarakau_basement_WGS84.h5 models/Rarakau.zarr --priority 0 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

springs_junction:
    @test -d models/SpringsJunction.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/SpringsJunction/SpringsJunction_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/SpringsJunction/SpringsJunction_basement_WGS84.h5 models/SpringsJunction.zarr --priority 22 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

tolaga_bay:
    @test -d models/TolagaBay.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/TolagaBay/TolagaBay_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/TolagaBay/TolagaBay_basement_WGS84.h5 models/TolagaBay.zarr --priority 8 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

waiapu:
    @test -d models/Waiapu.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Waiapu/Waiapu_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Waiapu/Waiapu_basement_WGS84.h5 models/Waiapu.zarr --priority 7 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

waikato_hauraki:
    @test -d models/WaikatoHauraki.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/WaikatoHauraki/WaikatoHauraki_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/WaikatoHauraki/WaikatoHauraki_basement_WGS84.h5 models/WaikatoHauraki.zarr --priority 37 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

wairarapa:
    @test -d models/Wairarapa.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Wairarapa/Wairarapa_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Wairarapa/Wairarapa_basement_WGS84.h5 models/Wairarapa.zarr --priority 15 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

waitaki:
    @test -d models/Waitaki.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Waitaki/Waitaki_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Waitaki/Waitaki_basement_WGS84.h5 models/Waitaki.zarr --priority 26 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

wakatipu:
    @test -d models/Wakatipu.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Wakatipu/Wakatipu_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Wakatipu/Wakatipu_basement_WGS84.h5 models/Wakatipu.zarr --priority 34 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

wanaka:
    @test -d models/Wanaka.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Wanaka/Wanaka_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Wanaka/Wanaka_basement_WGS84.h5 models/Wanaka.zarr --priority 36 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

westport:
    @test -d models/Westport.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Westport/Westport_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Westport/Westport_basement_WGS84.h5 models/Westport.zarr --priority 5 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile  --smoothing {{ smoothing }} --coastline {{ coastline }}

north_canterbury:
    @test -d models/NorthCanterbury.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/NorthCanterbury/NorthCanterbury_outline_WGS84_v19p1.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/NorthCanterbury/NorthCanterbury_basement_WGS84_v19p1.h5 models/NorthCanterbury.zarr --priority 45 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile 

wellington:
    @test -d models/Wellington.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Wellington/Wellington_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Wellington/Wellington_basement_WGS84_v21p8.h5 models/Wellington.zarr --priority 38 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile  --smoothing {{ smoothing }} --coastline {{ coastline }}

palmerston_north:
    @test -d models/PalmerstonNorth.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/PalmerstonNorth/PalmerstonNorth_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/PalmerstonNorth/PalmerstonNorth_pliocenetop_WGS84_v25p8.h5 models/PalmerstonNorth.zarr --priority 11 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/PalmerstonNorth_v1.fd_modfile  --smoothing {{ smoothing }} --coastline {{ coastline }}
    @test -d models/PalmerstonNorthPliocene.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/PalmerstonNorth/PalmerstonNorth_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/PalmerstonNorth/PalmerstonNorth_pliocenetop_WGS84_v25p8.h5 ${NZCVM_DATA_ROOT}/regional/PalmerstonNorth/PalmerstonNorth_basement_WGS84_v25p8.h5 models/PalmerstonNorthPliocene.zarr --priority 12 --rho 2120 --vp 2600 --vs 1100 --qs 55 --qp 110  --smoothing {{ smoothing }} --coastline {{ coastline }} --no-pad-top

omaio_bay:
    @for i in 1 2 3; do \
        test -d models/OmaioBay${i}.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/OmaioBay/OmaioBay_outline_WGS84_v22p3_${i}.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/OmaioBay/OmaioBay_basement_WGS84_v22p3.h5 models/OmaioBay${i}.zarr --priority 14 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile  --smoothing {{ smoothing }} --coastline {{ coastline }}; \
    done

nelson:
    @test -d models/Nelson.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Nelson/Nelson_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Nelson/Nelson_basement_WGS84_v25p5.h5 models/Nelson.zarr --priority 39 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile  --smoothing {{ smoothing }} --coastline {{ coastline }}

kaikoura:
    @test -d models/Kaikoura.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Kaikoura/Kaikoura_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Kaikoura/Kaikoura_basement_WGS84.h5 models/Kaikoura.zarr --priority 43 --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile  --smoothing {{ smoothing }} --coastline {{ coastline }}

[parallel]
basins: alexandra balclutha canterbury castle_hill cheviot collingwood dunedin gisborne greater_wellington hakataramea hanmer hawkes_bay kaikoura karamea mackenzie marlborough mosgiel murchison napier ne_otago nelson north_canterbury omaio_bay palmerston_north porirua queen_charlotte ranfurly rarakau southern_hawkes_bay southland springs_junction te_anau te_araroa tolaga_bay waiapu waikato_hauraki wairarapa waitaki wakatipu wanaka wellington west_coast westport whakatane whangaparoa

tomography := "uv run nzcvm tomography convert"
surface := "uv run nzcvm convert-tiff main"

ep2020:
    @test -f models/ep2020.zarr || {{ tomography }} ep2020.csv models/ep2020.zarr

# Compile every mesh's BVH once, so a run maps it instead of rebuilding it.
index:
    uv run nzcvm index build models/*.zarr

models: ep2020 basins index
    MODEL_PATH=$(realpath models) uv run pytest -svv tests/test_models.py
    zip -r models.zip models

vs30:
    @test -f resources/vs30.zarr || {{ surface }} ${VS30_TIFF} 1 resources/vs30.zarr --downsample 3 

pytest:
    SETUPTOOLS_RUST_CARGO_PROFILE=dev uv run --dev pytest -s tests
    SETUPTOOLS_RUST_CARGO_PROFILE=dev uv run --dev pytest --doctest-modules nzcvm/ -v

cargo:
    cargo test
    cargo test --features high_precision
    cargo test --doc

test: pytest cargo

ty:
    uv run ty check nzcvm

ruff:
    uv run ruff format
    uv run ruff check --select I --fix

clippy:
    cargo clippy -- -D warnings

lint: ty ruff clippy

# ---------------------------------------------------------------------------
# Synthetic data
#
# `just synthetic` builds a self-contained data root from closed-form fields
# (see nzcvm/synthetic.py), small enough to regenerate in seconds and complete
# enough to drive `nzcvm generate`. It needs no NZCVM_DATA_ROOT.
# ---------------------------------------------------------------------------

synthetic_root := "synthetic"
synthetic := "uv run nzcvm synthetic"
convert_surface := "uv run nzcvm surface convert"

# Every input `examples/synthetic.toml` reads.
synthetic: synthetic_dem synthetic_vs30 synthetic_coastline synthetic_models synthetic_index

synthetic_dem:
    {{ synthetic }} dem {{ synthetic_root }}/dem.h5
    {{ convert_surface }} {{ synthetic_root }}/dem.h5 {{ synthetic_root }}/dem.zarr

synthetic_vs30:
    {{ synthetic }} vs30 {{ synthetic_root }}/vs30.h5
    {{ convert_surface }} {{ synthetic_root }}/vs30.h5 {{ synthetic_root }}/vs30.zarr --scalar-key vs30 --no-flip

synthetic_coastline:
    {{ synthetic }} coastline {{ synthetic_root }}/coastline.wkb.gz

synthetic_tomography:
    {{ synthetic }} tomography {{ synthetic_root }}/tomography.csv
    {{ tomography }} {{ synthetic_root }}/tomography.csv {{ synthetic_root }}/models/tomography.zarr

# Priorities mirror the real basin recipes: lower wins over the tomography.
synthetic_basins: synthetic_dem
    {{ synthetic }} profile {{ synthetic_root }}/profile.fd_modfile
    @for basin in gully terrace; do \
        {{ synthetic }} basin $basin {{ synthetic_root }}; \
        {{ construct }} {{ synthetic_root }}/${basin}_outline.geojson {{ synthetic_root }}/dem.h5 {{ synthetic_root }}/dem.h5 {{ synthetic_root }}/${basin}_basement.h5 {{ synthetic_root }}/models/${basin}.zarr --priority 1 --vm-1d {{ synthetic_root }}/profile.fd_modfile; \
    done

synthetic_models: synthetic_tomography synthetic_basins

synthetic_index: synthetic_models
    uv run nzcvm index build {{ synthetic_root }}/models/*.zarr

clean_synthetic:
    rm -rf {{ synthetic_root }}
