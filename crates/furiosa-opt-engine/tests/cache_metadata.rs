use furiosa_opt_engine::decoder::{CacheSpec, Group, Kind, PageFormat, Pool};
use std::io;

fn pool(block: usize, bytes: usize) -> Pool {
    Pool {
        block,
        bytes,
        format: PageFormat::contiguous(vec![block, bytes / block], 1),
    }
}

fn spec() -> CacheSpec {
    CacheSpec {
        layer_count: 3,
        pools: vec![pool(16, 64), pool(16, 32)],
        groups: vec![
            Group {
                pool: 0,
                layers: vec![0, 2],
                logical_group: 0,
                kind: Kind::Paged,
                window: None,
            },
            Group {
                pool: 1,
                layers: vec![1],
                logical_group: 0,
                kind: Kind::Paged,
                window: None,
            },
        ],
    }
}

fn message(error: io::Error) -> String {
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    error.to_string()
}

#[test]
fn accepts_non_contiguous_layers_and_heterogeneous_engine_groups() {
    let cache = spec();
    cache.validate().unwrap();
    cache.validate_for_serving().unwrap();
}

#[test]
fn layer_storage_sizes_keep_the_signed_i32_boundary() {
    let cache = spec();
    assert_eq!(
        cache.layer_sizes(1 << 25).unwrap(),
        [1 << 31, 1 << 30, 1 << 31]
    );
    let error = cache.layer_sizes((1 << 25) + 1).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert_eq!(error.to_string(), "cache blocks pass I32 byte offsets");
    let error = cache.layer_sizes(usize::MAX).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert_eq!(error.to_string(), "cache blocks pass I32 byte offsets");
}

#[test]
fn rejects_duplicate_and_missing_layer_coverage() {
    let mut cache = spec();
    cache.groups[1].layers = vec![2];
    assert_eq!(
        message(cache.validate().unwrap_err()),
        "cache layer appears in more than one group"
    );

    let mut cache = spec();
    cache.layer_count += 1;
    assert_eq!(
        message(cache.validate().unwrap_err()),
        "cache spec does not cover every model layer"
    );
}

#[test]
fn rejects_inconsistent_logical_group_metadata() {
    let mut cache = spec();
    cache.pools[1].block = 8;
    cache.pools[1].format = PageFormat::contiguous(vec![8, 4], 1);
    assert_eq!(
        message(cache.validate().unwrap_err()),
        "logical cache groups disagree on block size or attention kind"
    );

    let mut cache = spec();
    cache.groups[1].logical_group = 2;
    assert_eq!(
        message(cache.validate().unwrap_err()),
        "logical cache group IDs are not contiguous"
    );
}

#[test]
fn rejects_invalid_sliding_window_metadata() {
    let mut cache = spec();
    cache.groups[0].window = Some(0);
    assert_eq!(
        message(cache.validate().unwrap_err()),
        "paged cache group has an empty sliding window"
    );

    let mut cache = spec();
    cache.groups[0].kind = Kind::State;
    cache.groups[0].window = Some(16);
    assert_eq!(
        message(cache.validate().unwrap_err()),
        "non-attention cache group has a sliding window"
    );
}

#[test]
fn rejects_overlapping_or_mis_sized_page_formats() {
    let mut cache = spec();
    cache.pools[0].format.strides = vec![1, 1];
    assert_eq!(
        message(cache.validate().unwrap_err()),
        "cache page strides overlap logical elements"
    );

    let mut cache = spec();
    cache.pools[0].format.strides = vec![8, 1];
    assert_eq!(
        message(cache.validate().unwrap_err()),
        "cache page format span differs from its pool size"
    );
}

#[test]
fn refuses_recurrent_and_encoder_cache_kinds_for_serving() {
    for kind in [Kind::State, Kind::Cross { length: 32 }] {
        let mut cache = spec();
        for group in &mut cache.groups {
            group.kind = kind;
        }
        let error = cache.validate_for_serving().unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Unsupported);
        assert_eq!(
            error.to_string(),
            "decoder serving supports paged cache groups only"
        );
    }
}
