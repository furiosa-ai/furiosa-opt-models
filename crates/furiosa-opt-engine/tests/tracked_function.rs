//! Accepted device work remains observable independently of the call's returned output.

use furiosa_opt_engine::{AsyncModule, Function, forward};
use furiosa_opt_models::Module;
use furiosa_opt_std::prelude::*;

axes![Width = 8];
type Rows = HbmTensor<f32, m![1], m![Width]>;

#[derive(DeviceSend)]
struct Echo;

impl<'x, 's> Module<&'x Rows, (), &'s mut ()> for Echo {
    type Output = Rows;

    fn forward(&self, device: &mut Device, _: (), _: &'s mut (), input: &'x Rows) -> Rows {
        input.to_hbm(&mut device.tdma)
    }
}

/// A completion survives dropping the returned output, and a later call still produces correct
/// rows. Ordinary acceptance-only calls preserve their existing result behavior.
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires one explicitly selected RNGD chip"]
async fn tracks_accepted_work_independently_of_output() {
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let expected: Vec<f32> = (0..Width::SIZE).map(|at| at as f32 - 3.0).collect();
    let input: Rows = HostTensor::<f32, m![Width]>::from_vec(expected.clone())
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    let function = Function::new(Echo, forward);
    let (output, work) = function
        .forward_tracked(&mut device, (), &mut (), &input)
        .unwrap();
    drop(output);
    work.ended().await.unwrap();
    assert_eq!(work.result(), Some(Ok(())));
    assert_eq!(work.wait(), Ok(()));

    let (output, work) = function
        .forward_tracked(&mut device, (), &mut (), &input)
        .unwrap();
    work.ended().await.unwrap();
    assert_eq!(
        output
            .to_host::<m![Width]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec(),
        expected
    );
    let output = function
        .forward(&mut device, (), &mut (), &input)
        .await
        .unwrap();
    assert_eq!(
        output
            .to_host::<m![Width]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec(),
        expected
    );
}
