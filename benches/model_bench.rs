use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, criterion_group, criterion_main,
};
use nalgebra::Point3;
use nzcvm::mesh::MeshModel;
use nzcvm::quality::Quality;
use nzcvm::real::Real;
use pprof::criterion::{Output, PProfProfiler};
use std::hint::black_box;

fn mock_quality() -> Quality {
    Quality {
        rho: 2500.0,
        vp: 3000.0,
        vs: 1500.0,
        qp: 100.0,
        qs: 50.0,
        alpha: 1.0,
    }
}

fn bench_mesh_queries(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("Mesh_Point_Query");
    group.plot_config(plot_config);

    for size in [160].iter() {
        let n = *size;
        let total_vertices = n * n * n;
        let vertices = (0..total_vertices)
            .map(|idx| {
                let i = idx / (n * n);
                let j = (idx / n) % n;
                let k = idx % n;
                Point3::new(i as Real, j as Real, k as Real)
            })
            .collect();

        let mesh = MeshModel::curvilinear_mesh(
            vertices,
            vec![mock_quality(); total_vertices],
            (n, n, n),
            |i, j, k| k + j * n + i * n * n,
        )
        .unwrap_or_else(|_| panic!("mesh construction failed"));

        // Each query point gets its own benchmark case.  Summing them inside a
        // single `b.iter` averaged the three together, which is the opposite of
        // isolating spatial bias: a regression affecting only deep-BVH lookups
        // would be diluted by two thirds.
        let queries = [
            ("near_origin", Point3::new(0.1, 0.1, 0.1)),
            (
                "centre",
                Point3::new(n as Real / 2.0, n as Real / 2.0, n as Real / 2.0),
            ),
            (
                "far_corner",
                Point3::new(n as Real - 1.1, n as Real - 1.1, n as Real - 1.1),
            ),
        ];

        for (name, query) in queries {
            group.bench_with_input(
                BenchmarkId::new(name, total_vertices),
                &mesh,
                |b, m: &MeshModel| {
                    // The result must be black-boxed, not just the input:
                    // discarding it lets LLVM elide the whole lookup, so the
                    // benchmark would happily "pass" if `query` returned `None`
                    // unconditionally.
                    b.iter(|| black_box(m.query(black_box(query))))
                },
            );
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .with_profiler(PProfProfiler::new(500, Output::Flamegraph(None)));
    targets =
        bench_mesh_queries,

}

criterion_main!(benches);
