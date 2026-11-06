use super::super::compact as tile;
use super::*;

pub(super) fn prepare<H: AxisName, I: AxisName, R: AxisName, const W: usize>(
    device: &mut Device,
    gate: &Panels<H, I>,
    up: &Panels<H, I>,
    down: &Down<I, H>,
    input: &Owned<H, R>,
    entry: Entry<H, I>,
    product: &mut Products<I, R>,
) -> down::Ahead<I, H> {
    let input = tile::input::<H, R, W>(device, input);
    let (g0, u0, d0) = entry;
    let g1 = gate.load(device, 1);
    let u1 = up.load(device, 1);
    let g0 = block::restore(device, g0.0, g0.1);
    let u0 = block::restore(device, u0.0, u0.1);
    let l0 = tile::lanes(device, &input, 0);
    let l1 = tile::lanes(device, &input, 1);
    let p0 = tile::pair::<H, I, R, W, _, _, _>(device, g0.view(), u0.view(), &l0);
    let g2 = gate.load(device, 2);
    let u2 = up.load(device, 2);
    let g1 = block::restore(device, g1.0, g1.1);
    let u1 = block::restore(device, u1.0, u1.1);
    let l2 = tile::lanes(device, &input, 2);
    let p1 = tile::pair::<H, I, R, W, _, _, _>(device, g1.view(), u1.view(), &l1);
    let g3 = gate.load(device, 3);
    let u3 = up.load(device, 3);
    let s1 = tile::add(device, p0.view(), p1.view());
    let g2 = block::restore(device, g2.0, g2.1);
    let u2 = block::restore(device, u2.0, u2.1);
    let l3 = tile::lanes(device, &input, 3);
    let p2 = tile::pair::<H, I, R, W, _, _, _>(device, g2.view(), u2.view(), &l2);
    let d1 = down.stage(device, 1);
    let s2 = tile::add(device, s1.view(), p2.view());
    let g3 = block::restore(device, g3.0, g3.1);
    let g3 = tile::single::<H, I, R, W, _, _, _>(device, g3.view(), &l3);
    let g = tile::term(device, &s2, 0);
    let g = tile::add(device, g.view(), g3.view());
    let g = block::finish(device, &gate.scales, g);
    let u3 = block::restore(device, u3.0, u3.1);
    let u3 = tile::single::<H, I, R, W, _, _, _>(device, u3.view(), &l3);
    let u = tile::term(device, &s2, 1);
    let u = tile::add(device, u.view(), u3.view());
    let u = block::finish(device, &up.scales, u);
    product::compact::<I, R, W>(device, &g, &u, product);
    (d0, d1)
}
