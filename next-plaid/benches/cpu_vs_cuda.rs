//! Benchmark: CPU vs CUDA for next-plaid search and indexing operations.
//!
//! Run with:
//!   cargo bench --bench cpu_vs_cuda --features cuda,openblas
//!
//! CPU only:
//!   cargo bench --bench cpu_vs_cuda --features openblas

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::time::Duration;

const DIM: usize = 128;
const NBITS: usize = 4;

fn random_embeddings(n: usize, dim: usize) -> Array2<f32> {
    let mut emb = Array2::random((n, dim), Uniform::new(-1.0f32, 1.0));
    for mut row in emb.axis_iter_mut(Axis(0)) {
        let norm = row.dot(&row).sqrt().max(1e-12);
        row /= norm;
    }
    emb
}

fn random_documents(num_docs: usize, dim: usize) -> Vec<Array2<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..num_docs)
        .map(|_| random_embeddings(rng.gen_range(80..=120), dim))
        .collect()
}

fn build_codec(centroids: &Array2<f32>) -> next_plaid::ResidualCodec {
    let dim = centroids.ncols();
    let n_options = 1usize << NBITS;
    let bucket_cutoffs: Vec<f32> = (1..n_options)
        .map(|i| (i as f32 / n_options as f32 - 0.5) * 2.0)
        .collect();
    let bucket_weights: Vec<f32> = (0..n_options)
        .map(|i| ((i as f32 + 0.5) / n_options as f32 - 0.5) * 2.0)
        .collect();
    next_plaid::ResidualCodec::new(
        NBITS,
        centroids.clone(),
        Array1::zeros(dim),
        Some(Array1::from_vec(bucket_cutoffs)),
        Some(Array1::from_vec(bucket_weights)),
    )
    .unwrap()
}

fn create_temp_index(num_docs: usize) -> (tempfile::TempDir, next_plaid::MmapIndex) {
    let docs = random_documents(num_docs, DIM);
    let tmp_dir = tempfile::tempdir().unwrap();
    let index_path = tmp_dir.path().join("index");
    std::fs::create_dir_all(&index_path).unwrap();
    let config = next_plaid::IndexConfig {
        nbits: NBITS,
        batch_size: 50_000,
        seed: Some(42),
        kmeans_niters: 2,
        max_points_per_centroid: 256,
        n_samples_kmeans: None,
        start_from_scratch: 0,
        force_cpu: true,
    };
    next_plaid::index::create_index_with_kmeans_files(&docs, index_path.to_str().unwrap(), &config)
        .unwrap();
    let index = next_plaid::MmapIndex::load(index_path.to_str().unwrap()).unwrap();
    (tmp_dir, index)
}

// --- 1. compress_into_codes ---

fn bench_compress_into_codes(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress_into_codes");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);

    for &(n_emb, n_c) in &[(10_000, 512), (100_000, 1024)] {
        let embeddings = random_embeddings(n_emb, DIM);
        let centroids = random_embeddings(n_c, DIM);
        let codec = build_codec(&centroids);
        let label = format!("{}emb_{}c", n_emb, n_c);

        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            b.iter(|| codec.compress_into_codes_cpu(&embeddings));
        });

        #[cfg(feature = "cuda")]
        if let Some(ctx) = next_plaid::cuda::get_global_context() {
            let cv = centroids.view();
            let ev = embeddings.view();
            group.bench_with_input(BenchmarkId::new("cuda", &label), &(), |b, _| {
                b.iter(|| {
                    next_plaid::cuda::compress_into_codes_cuda_batched(ctx, &ev, &cv, None).unwrap()
                });
            });
        }
    }
    group.finish();
}

// --- 2. compress_and_residuals ---

fn bench_compress_and_residuals(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress_and_residuals");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);

    let (n_emb, n_c) = (100_000, 1024);
    let embeddings = random_embeddings(n_emb, DIM);
    let centroids = random_embeddings(n_c, DIM);
    let codec = build_codec(&centroids);
    let label = format!("{}emb_{}c", n_emb, n_c);

    group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
        b.iter(|| {
            let codes = codec.compress_into_codes_cpu(&embeddings);
            let cv = codec.centroids_view();
            let mut residuals = embeddings.clone();
            for (i, mut row) in residuals.axis_iter_mut(Axis(0)).enumerate() {
                let centroid = cv.row(codes[i]);
                row.iter_mut()
                    .zip(centroid.iter())
                    .for_each(|(r, c)| *r -= c);
            }
            (codes, residuals)
        });
    });

    #[cfg(feature = "cuda")]
    if let Some(ctx) = next_plaid::cuda::get_global_context() {
        let cv = centroids.view();
        let ev = embeddings.view();
        group.bench_with_input(BenchmarkId::new("cuda", &label), &(), |b, _| {
            b.iter(|| {
                next_plaid::cuda::compress_and_residuals_cuda_batched(ctx, &ev, &cv, None).unwrap()
            });
        });
    }
    group.finish();
}

// --- 3. MaxSim scoring (single pair) ---

fn bench_maxsim(c: &mut Criterion) {
    let mut group = c.benchmark_group("maxsim_scoring");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(20);

    for &(qt, dt) in &[
        (32, 128),  // 4K — small
        (32, 512),  // 16K — medium
        (32, 4096), // 131K — large
        (64, 4096), // 262K — very large
    ] {
        let query = random_embeddings(qt, DIM);
        let doc = random_embeddings(dt, DIM);
        let label = format!("{}q_{}d", qt, dt);

        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            b.iter(|| next_plaid::maxsim::maxsim_score(&query.view(), &doc.view()));
        });
    }
    group.finish();
}

// --- 4. Batch MaxSim (exact scoring phase simulation) ---

fn bench_batch_maxsim(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_maxsim");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);

    let query = random_embeddings(32, DIM);
    let n_docs = 100;
    let docs: Vec<Array2<f32>> = (0..n_docs).map(|_| random_embeddings(100, DIM)).collect();
    let label = format!("{}docs_32q_100d", n_docs);

    group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
        b.iter(|| {
            docs.iter()
                .map(|d| next_plaid::maxsim::maxsim_score(&query.view(), &d.view()))
                .collect::<Vec<_>>()
        });
    });
    group.finish();
}

// --- 5. Index creation ---

fn bench_index_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_creation");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    let num_docs = 500;
    let docs = random_documents(num_docs, DIM);
    let kmeans_config = next_plaid::ComputeKmeansConfig {
        kmeans_niters: 2,
        max_points_per_centroid: 256,
        seed: 42,
        n_samples_kmeans: None,
        num_partitions: None,
        force_cpu: true,
    };
    let centroids = next_plaid::compute_kmeans(&docs, &kmeans_config).unwrap();

    group.bench_function("cpu_500docs", |b| {
        b.iter(|| {
            let tmp = tempfile::tempdir().unwrap();
            let p = tmp.path().join("idx");
            std::fs::create_dir_all(&p).unwrap();
            let cfg = next_plaid::IndexConfig {
                force_cpu: true,
                start_from_scratch: 0,
                seed: Some(42),
                ..Default::default()
            };
            next_plaid::index::create_index_files(
                &docs,
                centroids.clone(),
                p.to_str().unwrap(),
                &cfg,
            )
            .unwrap()
        });
    });

    #[cfg(feature = "cuda")]
    if next_plaid::cuda::get_global_context().is_some() {
        group.bench_function("cuda_500docs", |b| {
            b.iter(|| {
                let tmp = tempfile::tempdir().unwrap();
                let p = tmp.path().join("idx");
                std::fs::create_dir_all(&p).unwrap();
                let cfg = next_plaid::IndexConfig {
                    force_cpu: false,
                    start_from_scratch: 0,
                    seed: Some(42),
                    ..Default::default()
                };
                next_plaid::index::create_index_files(
                    &docs,
                    centroids.clone(),
                    p.to_str().unwrap(),
                    &cfg,
                )
                .unwrap()
            });
        });
    }
    group.finish();
}

// --- 6. Search pipeline ---

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_pipeline");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);

    let (_tmp, index) = create_temp_index(500);
    let query = random_embeddings(32, DIM);
    let params = next_plaid::SearchParameters {
        n_full_scores: 128,
        top_k: 10,
        n_ivf_probe: 8,
        centroid_score_threshold: Some(0.4),
        ..Default::default()
    };

    group.bench_function("500docs_32q", |b| {
        b.iter(|| next_plaid::search::search_one_mmap(&index, &query, &params, None).unwrap());
    });
    group.finish();
}

// --- 7. Centroid scoring GEMM ---

fn bench_centroid_scoring(c: &mut Criterion) {
    let mut group = c.benchmark_group("centroid_scoring_gemm");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(20);

    let query = random_embeddings(32, DIM);
    let centroids = random_embeddings(1024, DIM);

    group.bench_function("cpu_32q_1024c", |b| {
        b.iter(|| query.dot(&centroids.t()));
    });

    #[cfg(feature = "cuda")]
    if let Some(ctx) = next_plaid::cuda::get_global_context() {
        group.bench_function("cuda_32q_1024c", |b| {
            b.iter(|| {
                next_plaid::cuda::compress_into_codes_cuda_batched(
                    ctx,
                    &query.view(),
                    &centroids.view(),
                    None,
                )
                .unwrap()
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_compress_into_codes,
    bench_compress_and_residuals,
    bench_maxsim,
    bench_batch_maxsim,
    bench_index_creation,
    bench_search,
    bench_centroid_scoring,
);
criterion_main!(benches);
