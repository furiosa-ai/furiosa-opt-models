use super::*;

impl<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, Capacity: AxisName, W, S>
    Op<Begin<Live<Halves<Intermediate, Rows>, Capacity>>>
    for fp8::Rows<m![Intermediate], m![Hidden], W, S>
where
    W: Borrow<HbmTensor<f8e4m3, Chip, Stored<Hidden, Intermediate>>>,
{
    type Output = Staging<Hidden, Intermediate, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        Begin(product): Begin<Live<Halves<Intermediate, Rows>, Capacity>>,
    ) -> Self::Output {
        let (products, panels, scales) = spill(&mut *context, product);
        let first = self.forward(&mut *context, Begin(0));
        Staging {
            products,
            panels,
            scales,
            first,
        }
    }
}

impl<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, Capacity: AxisName, W, S>
    Op<
        Begin<(
            Live<Halves<Intermediate, Rows>, Capacity>,
            Down<Hidden, Intermediate>,
        )>,
    > for fp8::Rows<m![Intermediate], m![Hidden], W, S>
{
    type Output = Staging<Hidden, Intermediate, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        Begin((product, first)): Begin<(
            Live<Halves<Intermediate, Rows>, Capacity>,
            Down<Hidden, Intermediate>,
        )>,
    ) -> Self::Output {
        let (products, panels, scales) = spill(context, product);
        Staging {
            products,
            panels,
            scales,
            first,
        }
    }
}

impl<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, Capacity: AxisName, W, S>
    Op<Live<Staging<Hidden, Intermediate, Rows>, Capacity>>
    for fp8::Rows<m![Intermediate], m![Hidden], W, S>
where
    W: Borrow<HbmTensor<f8e4m3, Chip, Stored<Hidden, Intermediate>>>,
    S: Borrow<HbmTensor<f32, Chip, m![Hidden]>>,
{
    type Output = Owned<Hidden, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        staging: Live<Staging<Hidden, Intermediate, Rows>, Capacity>,
    ) -> Self::Output {
        const { assert!(<m![{ Out<Intermediate> } / 64]>::SIZE == Panels::SIZE) };
        let (live, rows) = (staging.count::<axis::Rows<128>>(), staging.rows());
        let Staging {
            products,
            panels: (h0, h1, h2),
            scales,
            first,
        } = staging.into_value();
        // SAFETY: the store's row digits are the span's rows in order.
        let products: HbmTensor<bf16, Chip, Chunks<Intermediate, Rows>> =
            unsafe { products.reshape() };
        let channels = self.1.borrow().to_dm(&mut context.tdma);
        let stored = (&products, &scales);
        let input0 = landed(&mut *context, stored, (0, &h0), (live, rows));
        let input1 = landed(&mut *context, stored, (1, &h1), (live, rows));
        let second = self.forward(&mut *context, Begin(1));
        let input2 = landed(&mut *context, stored, (2, &h2), (live, rows));
        let third = |context: &mut Device| self.forward(context, Begin(2));
        let inputs = (input0, input1, input2);
        projected(
            context,
            ((first, second), third),
            (inputs, channels),
            (live, rows),
        )
    }
}

impl<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, Capacity: AxisName, W, S>
    Op<
        Live<
            (
                Staging<Hidden, Intermediate, Rows>,
                (Down<Hidden, Intermediate>, Down<Hidden, Intermediate>),
            ),
            Capacity,
        >,
    > for fp8::Rows<m![Intermediate], m![Hidden], W, S>
where
    S: Borrow<HbmTensor<f32, Chip, m![Hidden]>>,
{
    type Output = Ready<Hidden, Intermediate, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        staged: Live<
            (
                Staging<Hidden, Intermediate, Rows>,
                (Down<Hidden, Intermediate>, Down<Hidden, Intermediate>),
            ),
            Capacity,
        >,
    ) -> Self::Output {
        const { assert!(<m![{ Out<Intermediate> } / 64]>::SIZE == Panels::SIZE) };
        let (live, rows) = (staged.count::<axis::Rows<128>>(), staged.rows());
        let (staging, (second, third)) = staged.into_value();
        let Staging {
            products,
            panels: (h0, h1, h2),
            scales,
            first,
        } = staging;
        // SAFETY: the store's row digits are the span's rows in order.
        let products: HbmTensor<bf16, Chip, Chunks<Intermediate, Rows>> =
            unsafe { products.reshape() };
        let channels = self.1.borrow().to_dm(&mut context.tdma);
        let stored = (&products, &scales);
        let inputs = (
            landed(&mut *context, stored, (0, &h0), (live, rows)),
            landed(&mut *context, stored, (1, &h1), (live, rows)),
            landed(&mut *context, stored, (2, &h2), (live, rows)),
        );
        Ready {
            weights: (first, second, third),
            inputs,
            channels,
        }
    }
}

impl<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, Capacity: AxisName, W, S>
    Op<Live<Ready<Hidden, Intermediate, Rows>, Capacity>>
    for fp8::Rows<m![Intermediate], m![Hidden], W, S>
{
    type Output = Owned<Hidden, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        ready: Live<Ready<Hidden, Intermediate, Rows>, Capacity>,
    ) -> Self::Output {
        let (live, rows) = (ready.count::<axis::Rows<128>>(), ready.rows());
        let Ready {
            weights: (first, second, third),
            inputs,
            channels,
        } = ready.into_value();
        let third = |_: &mut Device| third;
        projected(
            context,
            ((first, second), third),
            (inputs, channels),
            (live, rows),
        )
    }
}
