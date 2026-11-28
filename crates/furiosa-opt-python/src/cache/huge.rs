use std::io;

use memmap2::{Advice, MmapMut};

const SIZE: usize = 2 << 20;
const THREADS: usize = 8;
const NONE: libc::c_ulong = 0;

/// Faults huge pages before allocating the pinned slots that consume them.
pub(super) struct Pages {
    _region: MmapMut,
}

impl Pages {
    pub(super) fn fault(bytes: usize) -> io::Result<Self> {
        let inherited = disabled()?;
        disable(0)?;
        let region = Self::touch(bytes);
        let restore = disable(inherited);
        match (region, restore) {
            (Ok(region), Ok(())) => Ok(Self { _region: region }),
            (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
            (Err(error), Err(restore)) => Err(io::Error::new(
                error.kind(),
                format!("{error}; restoring the inherited THP setting also failed: {restore}"),
            )),
        }
    }

    fn touch(bytes: usize) -> io::Result<MmapMut> {
        let mut region = MmapMut::map_anon(bytes + SIZE)?;
        region.advise(Advice::HugePage)?;
        let start = region.as_ptr().addr().next_multiple_of(SIZE) - region.as_ptr().addr();
        let share = bytes.div_ceil(THREADS).next_multiple_of(SIZE);
        std::thread::scope(|scope| {
            for part in region[start..start + bytes].chunks_mut(share) {
                std::thread::Builder::new().spawn_scoped(scope, || {
                    for page in part.chunks_mut(SIZE) {
                        // SAFETY: each thread exclusively faults its borrowed part of the mapping.
                        unsafe { page.as_mut_ptr().write_volatile(1) };
                    }
                })?;
            }
            Ok::<_, io::Error>(())
        })?;
        Ok(region)
    }
}

fn disabled() -> io::Result<libc::c_ulong> {
    // SAFETY: PR_GET_THP_DISABLE only reads this process's setting.
    match unsafe { libc::prctl(libc::PR_GET_THP_DISABLE, NONE, NONE, NONE, NONE) } {
        0 => Ok(0),
        1 => Ok(1),
        _ => Err(io::Error::last_os_error()),
    }
}

fn disable(flag: libc::c_ulong) -> io::Result<()> {
    // SAFETY: PR_SET_THP_DISABLE changes this process's THP fault policy.
    match unsafe { libc::prctl(libc::PR_SET_THP_DISABLE, flag, NONE, NONE, NONE) } {
        0 => Ok(()),
        _ => Err(io::Error::last_os_error()),
    }
}
