//! Dense attention over caller-supplied visible keys.

use furiosa_opt_std::prelude::*;

#[derive(Clone, Copy, DeviceSend)]
pub struct Compiled<const EPSILON: usize>;

impl<const BITS: usize> From<Compiled<BITS>> for f32 {
    fn from(_: Compiled<BITS>) -> Self {
        f32::from_bits(
            const {
                assert!(BITS <= u32::MAX as usize);
                BITS as u32
            },
        )
    }
}

/// Head rows of a row sequence in HBM.
pub(crate) mod heads {

    use furiosa_opt_std::prelude::*;
    use std::marker::PhantomData;

    /// The heads of a row of `Width` features, `Head` features each.
    #[derive(Clone, Debug)]
    pub struct Heads<Width, Head>(PhantomData<(Width, Head)>);

    impl<Width: AxisName, Head: AxisName> AxisName for Heads<Width, Head> {
        const NAME: Ident = Ident::new("Heads");

        const SIZE: usize = {
            assert!(Head::SIZE > 0 && Width::SIZE % Head::SIZE == 0);
            assert!(Width::SIZE / Head::SIZE <= 64);
            Width::SIZE / Head::SIZE
        };
    }

    pub(crate) type H<Width, Head> = Symbol<Heads<Width, Head>>;

    /// Head rows a tile at a time, row-major or head-major, each slice 512 bytes of a row; the
    /// next tile lands, and the last one stores, while the current one computes.
    pub(crate) mod tiles {

        use furiosa_opt_std::prelude::*;
        use std::marker::PhantomData;

        /// The heads a slice holds of a row.
        #[derive(Clone, Debug)]
        pub struct Kept<Width, Head>(PhantomData<(Width, Head)>);

        impl<Width: AxisName, Head: AxisName> AxisName for Kept<Width, Head> {
            const NAME: Ident = Ident::new("HeadsKept");
            const SIZE: usize = {
                let kept = if Head::SIZE >= 256 {
                    1
                } else {
                    256 / Head::SIZE
                };
                assert!((Width::SIZE / Head::SIZE) % kept == 0);
                kept
            };
        }

        /// The slices of a row group: its head groups, padded.
        #[derive(Clone, Debug)]
        pub struct Slots<Width, Head>(PhantomData<(Width, Head)>);

        impl<Width: AxisName, Head: AxisName> AxisName for Slots<Width, Head> {
            const NAME: Ident = Ident::new("HeadSlots");
            const SIZE: usize = {
                let groups = Width::SIZE / Head::SIZE / Kept::<Width, Head>::SIZE;
                assert!(groups > 0 && groups <= 64);
                groups.next_power_of_two()
            };
        }

        /// The rows a slice holds.
        #[derive(Clone, Debug)]
        pub struct Depth<Width, Head>(PhantomData<(Width, Head)>);

        impl<Width: AxisName, Head: AxisName> AxisName for Depth<Width, Head> {
            const NAME: Ident = Ident::new("HeadDepth");
            const SIZE: usize = if Head::SIZE <= 64 { 8 } else { 4 };
        }

        /// The rows of a tile.
        #[derive(Clone, Debug)]
        pub struct Strip<Width, Head>(PhantomData<(Width, Head)>);

        impl<Width: AxisName, Head: AxisName> AxisName for Strip<Width, Head> {
            const NAME: Ident = Ident::new("HeadStrip");
            const SIZE: usize = 2 * (256 / Slots::<Width, Head>::SIZE) * Depth::<Width, Head>::SIZE;
        }

        /// The rows of a tile one cluster holds.
        #[derive(Clone, Debug)]
        pub struct Group<Width, Head>(PhantomData<(Width, Head)>);

        impl<Width: AxisName, Head: AxisName> AxisName for Group<Width, Head> {
            const NAME: Ident = Ident::new("HeadGroup");
            const SIZE: usize = Strip::<Width, Head>::SIZE / 2;
        }

        /// The tiles of a row sequence.
        #[derive(Clone, Debug)]
        pub struct Strips<Count, Width, Head, P = (), Chips = m![1], Q = ()>(
            pub PhantomData<(Count, Width, Head, Chips, Q)>,
            pub P,
        );

        impl<Count: AxisName, Width: AxisName, Head: AxisName> AxisName for Strips<Count, Width, Head> {
            const NAME: Ident = Ident::new("HeadStrips");
            const SIZE: usize = {
                let strip = Strip::<Width, Head>::SIZE;
                assert!(Count::SIZE % strip == 0);
                Count::SIZE / strip
            };
        }

        use super::H;
        pub(crate) type K<Width, Head> = Symbol<Kept<Width, Head>>;
        pub(crate) type P<Width, Head> = Symbol<Slots<Width, Head>>;
        pub(crate) type D<Width, Head> = Symbol<Depth<Width, Head>>;
        pub(crate) type T<Width, Head> = Symbol<Strip<Width, Head>>;
        pub(crate) type G<Width, Head> = Symbol<Group<Width, Head>>;
        pub(crate) type S<Count, Width, Head> = Symbol<Strips<Count, Width, Head>>;
        pub(crate) type Banded<Count, Width, Head> =
            m![{ S<Count, Width, Head> }, { T<Width, Head> }, { H<Width, Head> }, Head];
        pub(crate) type Half<Width, Head> = m![{ T<Width, Head> } / ({ G<Width, Head> })];
        pub(crate) type Slices<Width, Head> = m![
            { H<Width, Head> } / ({ K<Width, Head> }) # ({ P<Width, Head> }),
            { T<Width, Head> } % ({ G<Width, Head> }) / ({ D<Width, Head> })
        ];
        /// A slice's rows, then the heads it holds of each: the order row-major rows move in.
        pub(crate) type Rows<Width, Head> = m![
            { T<Width, Head> } % ({ D<Width, Head> }),
            { H<Width, Head> } % ({ K<Width, Head> })
        ];
        /// A slice's heads, then each head's rows: the order head-major rows move in.
        pub(crate) type Columns<Width, Head> = m![
            { H<Width, Head> } % ({ K<Width, Head> }),
            { T<Width, Head> } % ({ D<Width, Head> })
        ];
        /// A tile, its rows' heads in a slice in order `Q`.
        pub(crate) type Tile<E, Width, Head, Chips, Q> =
            DmTensor<E, Chips, Half<Width, Head>, Slices<Width, Head>, m![{ Q }, Head]>;

        /// Head rows in HBM: row-major, every row its heads in order, or head-major, every head
        /// its rows in order.
        pub(crate) trait Layout<Count: AxisName, Width: AxisName, Head: AxisName, Chips: M> {
            /// The order a slice holds its rows' heads in: the order they move in.
            type Order: M;

            fn new() -> Self;

            /// Tile `at`, landed into `tile`.
            fn land(
                &self,
                device: &mut Device,
                at: usize,
                tile: &mut Tile<bf16, Width, Head, Chips, Self::Order>,
            );

            /// `tile` stored as tile `at`.
            fn store(
                &mut self,
                device: &mut Device,
                tile: &Tile<bf16, Width, Head, Chips, Self::Order>,
                at: usize,
            );
        }

        /// Head-major rows of `Width` features, `Head` features a head.
        pub(crate) type Major<Count, Width, Head, Chips = m![1]> = HbmTensor<
            bf16,
            Chips,
            m![Width / ({ Symbol<Head> }), Count, Width % ({ Symbol<Head> })],
        >;
        type Headed<Count, Width, Head> =
            m![{ H<Width, Head> }, { S<Count, Width, Head> }, { T<Width, Head> }, Head];

        impl<Count: AxisName, Width: AxisName, Head: AxisName, Chips: M>
            Layout<Count, Width, Head, Chips> for HbmTensor<bf16, Chips, m![Count, Width]>
        {
            type Order = Rows<Width, Head>;

            fn new() -> Self {
                HbmTensor::new()
            }

            fn land(
                &self,
                device: &mut Device,
                at: usize,
                tile: &mut Tile<bf16, Width, Head, Chips, Self::Order>,
            ) {
                // SAFETY: the tiles split the rows in order and a row is its heads in order.
                unsafe { self.view().reshape::<Chips, Banded<Count, Width, Head>>() }
                    .tile::<m![{ S<Count, Width, Head> }], 1, m![
                        { S<Count, Width, Head> } = 1 # ({ S<Count, Width, Head> }),
                        { T<Width, Head> },
                        { H<Width, Head> },
                        Head
                    ]>(at)
                    .to_dm_view(&mut device.tdma, tile.view_mut());
            }

            fn store(
                &mut self,
                device: &mut Device,
                tile: &Tile<bf16, Width, Head, Chips, Self::Order>,
                at: usize,
            ) {
                tile.view().to_hbm_view(
                    &mut device.tdma,
                    // SAFETY: as for `land`.
                    unsafe {
                        self.view_mut()
                            .reshape::<Chips, Banded<Count, Width, Head>>()
                    }
                    .tile::<m![{ S<Count, Width, Head> }], 1, m![
                        { S<Count, Width, Head> } = 1 #{!} ({ S<Count, Width, Head> }),
                        { T<Width, Head> },
                        { H<Width, Head> },
                        Head
                    ]>(at),
                );
            }
        }

        impl<Count: AxisName, Width: AxisName, Head: AxisName, Chips: M>
            Layout<Count, Width, Head, Chips> for Major<Count, Width, Head, Chips>
        {
            type Order = Columns<Width, Head>;

            fn new() -> Self {
                HbmTensor::new()
            }

            fn land(
                &self,
                device: &mut Device,
                at: usize,
                tile: &mut Tile<bf16, Width, Head, Chips, Self::Order>,
            ) {
                // SAFETY: the tiles split each head's rows in order; a head is its features in
                // order.
                unsafe { self.view().reshape::<Chips, Headed<Count, Width, Head>>() }
                    .tile::<m![{ S<Count, Width, Head> }], 1, m![
                        { H<Width, Head> },
                        { S<Count, Width, Head> } = 1 # ({ S<Count, Width, Head> }),
                        { T<Width, Head> },
                        Head
                    ]>(at)
                    .to_dm_view(&mut device.tdma, tile.view_mut());
            }

            fn store(
                &mut self,
                device: &mut Device,
                tile: &Tile<bf16, Width, Head, Chips, Self::Order>,
                at: usize,
            ) {
                tile.view().to_hbm_view(
                    &mut device.tdma,
                    // SAFETY: as for `land`.
                    unsafe {
                        self.view_mut()
                            .reshape::<Chips, Headed<Count, Width, Head>>()
                    }
                    .tile::<m![{ S<Count, Width, Head> }], 1, m![
                        { H<Width, Head> },
                        { S<Count, Width, Head> } = 1 #{!} ({ S<Count, Width, Head> }),
                        { T<Width, Head> },
                        Head
                    ]>(at),
                );
            }
        }

        /// One pass over a tile of head rows.
        pub(crate) trait Pass<Width: AxisName, Head: AxisName, Chips: M, Q: M> {
            /// Tile `at`, landed, through the pass into `output`.
            fn apply(
                &self,
                device: &mut Device,
                tile: &Tile<bf16, Width, Head, Chips, Q>,
                at: usize,
                output: &mut Tile<bf16, Width, Head, Chips, Q>,
            );
        }

        /// Every tile of `input` through `pass` into `output`, the next tile landing while one
        /// computes and the tile before it storing.
        impl<Count: AxisName, Width: AxisName, Head: AxisName, Chips: M, Q: M, L, P>
            crate::Module<(&mut Device, (&L, &mut L))> for Strips<Count, Width, Head, P, Chips, Q>
        where
            L: Layout<Count, Width, Head, Chips, Order = Q>,
            P: Pass<Width, Head, Chips, Q>,
        {
            type Output = ();

            fn forward(&self, (device, (input, output)): (&mut Device, (&L, &mut L))) {
                let pass = &self.1;
                let tiles = Strips::<Count, Width, Head>::SIZE;
                let (mut first, mut second): (
                    Tile<bf16, Width, Head, Chips, Q>,
                    Tile<bf16, Width, Head, Chips, Q>,
                ) = (DmTensor::new(), DmTensor::new());
                let (mut kept, mut held): (
                    Tile<bf16, Width, Head, Chips, Q>,
                    Tile<bf16, Width, Head, Chips, Q>,
                ) = (DmTensor::new(), DmTensor::new());
                input.land(device, 0, &mut first);
                if const { Strips::<Count, Width, Head>::SIZE > 1 } {
                    input.land(device, 1, &mut second);
                }
                for pair in 0..tiles / 2 {
                    let at = 2 * pair;
                    pass.apply(device, &first, at, &mut kept);
                    input.land(device, (at + 2).min(tiles - 1), &mut first);
                    output.store(device, &kept, at);
                    pass.apply(device, &second, at + 1, &mut held);
                    input.land(device, (at + 3).min(tiles - 1), &mut second);
                    output.store(device, &held, at + 1);
                }
                if const { Strips::<Count, Width, Head>::SIZE % 2 == 1 } {
                    pass.apply(device, &first, tiles - 1, &mut kept);
                    output.store(device, &kept, tiles - 1);
                }
            }
        }

        /// One pass over the same tile of two row sequences, so what the tile's rows share lands
        /// once for both.
        pub(crate) trait Both<Width: AxisName, Head: AxisName, Chips: M, Q: M> {
            /// Tile `at` of both sequences, landed, through the pass into the outputs.
            fn apply(
                &self,
                device: &mut Device,
                tiles: (
                    &Tile<bf16, Width, Head, Chips, Q>,
                    &Tile<bf16, Width, Head, Chips, Q>,
                ),
                at: usize,
                outputs: (
                    &mut Tile<bf16, Width, Head, Chips, Q>,
                    &mut Tile<bf16, Width, Head, Chips, Q>,
                ),
            );
        }

        impl<Count: AxisName, Width: AxisName, Head: AxisName, Chips: M, Q: M, L, P>
            crate::Module<(&mut Device, ((&L, &L), (&mut L, &mut L)))>
            for Strips<Count, Width, Head, P, Chips, Q>
        where
            L: Layout<Count, Width, Head, Chips, Order = Q>,
            P: Both<Width, Head, Chips, Q>,
        {
            type Output = ();

            fn forward(
                &self,
                (device, (inputs, outputs)): (&mut Device, ((&L, &L), (&mut L, &mut L))),
            ) {
                let pass = &self.1;
                type Two<Width, Head, Chips, Q> = (
                    Tile<bf16, Width, Head, Chips, Q>,
                    Tile<bf16, Width, Head, Chips, Q>,
                );
                let tiles = Strips::<Count, Width, Head>::SIZE;
                let land_both =
                    |device: &mut Device, at: usize, both: &mut Two<Width, Head, Chips, Q>| {
                        inputs.0.land(device, at, &mut both.0);
                        inputs.1.land(device, at, &mut both.1);
                    };
                let (first_out, second_out) = outputs;
                let mut store_both =
                    |device: &mut Device, both: &Two<Width, Head, Chips, Q>, at: usize| {
                        first_out.store(device, &both.0, at);
                        second_out.store(device, &both.1, at);
                    };
                let mut first: Two<Width, Head, Chips, Q> = (DmTensor::new(), DmTensor::new());
                let mut second: Two<Width, Head, Chips, Q> = (DmTensor::new(), DmTensor::new());
                let mut kept: Two<Width, Head, Chips, Q> = (DmTensor::new(), DmTensor::new());
                let mut held: Two<Width, Head, Chips, Q> = (DmTensor::new(), DmTensor::new());
                land_both(device, 0, &mut first);
                if const { Strips::<Count, Width, Head>::SIZE > 1 } {
                    land_both(device, 1, &mut second);
                }
                for pair in 0..tiles / 2 {
                    let at = 2 * pair;
                    pass.apply(device, (&first.0, &first.1), at, (&mut kept.0, &mut kept.1));
                    land_both(device, (at + 2).min(tiles - 1), &mut first);
                    store_both(device, &kept, at);
                    pass.apply(
                        device,
                        (&second.0, &second.1),
                        at + 1,
                        (&mut held.0, &mut held.1),
                    );
                    land_both(device, (at + 3).min(tiles - 1), &mut second);
                    store_both(device, &held, at + 1);
                }
                if const { Strips::<Count, Width, Head>::SIZE % 2 == 1 } {
                    pass.apply(
                        device,
                        (&first.0, &first.1),
                        tiles - 1,
                        (&mut kept.0, &mut kept.1),
                    );
                    store_both(device, &kept, tiles - 1);
                }
            }
        }
    }
}
