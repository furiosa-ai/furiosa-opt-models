use super::{Disabled, Literal};
use serde::Deserialize;
use std::io;

#[derive(Clone)]
pub struct Config {
    alpha: Vec<f32>,
    terminal: f32,
    limit: f32,
}

impl TryFrom<Hf> for Config {
    type Error = io::Error;

    fn try_from(hf: Hf) -> io::Result<Self> {
        let invalid = |reason: String| io::Error::new(io::ErrorKind::InvalidData, reason);
        let Hf {
            prediction_type: None | Some(Literal),
            thresholding: None | Some(Disabled),
            rescale_betas_zero_snr: None | Some(Disabled),
            ..
        } = hf;
        let count = hf.num_train_timesteps.unwrap_or(1000);
        if count == 0 {
            return Err(invalid(
                "num_train_timesteps: 0, a schedule has a timestep".into(),
            ));
        }
        let number = |field: &str, value: Option<f64>, default: f64| {
            let value = value.unwrap_or(default);
            match value.is_finite() {
                true => Ok(value),
                false => Err(invalid(format!("{field}: {value}, not a finite number"))),
            }
        };
        let beta: Vec<f32> = match hf.trained_betas {
            Some(values) if values.len() != count => {
                let betas = values.len();
                return Err(invalid(format!(
                    "trained_betas: {betas} betas for {count} timesteps"
                )));
            }
            Some(values) => values.into_iter().map(|value| value as f32).collect(),
            None => {
                let start = number("beta_start", hf.beta_start, 0.0001)?;
                let end = number("beta_end", hf.beta_end, 0.02)?;
                let scaled = match hf.beta_schedule.as_deref().unwrap_or("linear") {
                    "linear" => false,
                    "scaled_linear" => true,
                    other => {
                        return Err(io::Error::new(
                            io::ErrorKind::Unsupported,
                            format!("beta_schedule: {other:?}, DDIM runs linear or scaled_linear"),
                        ));
                    }
                };
                let (start, end) = if scaled {
                    (start.sqrt(), end.sqrt())
                } else {
                    (start, end)
                };
                (0..count)
                    .map(|i| {
                        let value = (start
                            + (end - start) * i as f64 / count.saturating_sub(1).max(1) as f64)
                            as f32;
                        if scaled { value * value } else { value }
                    })
                    .collect()
            }
        };
        let outside = |&(_, &b): &(usize, &f32)| !b.is_finite() || b <= 0.0 || b >= 1.0;
        if let Some((step, beta)) = beta.iter().enumerate().find(outside) {
            return Err(invalid(format!(
                "beta {beta} at timestep {step}, outside (0, 1)"
            )));
        }
        // Training coefficients are fixed parameters, prepared once before placement.
        let alpha: Vec<_> = beta
            .into_iter()
            .scan(1.0_f32, |product, beta| {
                *product *= 1.0 - beta;
                Some(*product)
            })
            .collect();
        if let Some(step) = alpha.iter().position(|&a| a <= 0.0 || !a.is_finite()) {
            return Err(invalid(format!(
                "the alpha product underflows FP32 at timestep {step}"
            )));
        }
        let terminal = if hf.set_alpha_to_one.unwrap_or(true) {
            1.0
        } else {
            alpha[0]
        };
        let limit = match hf.clip_sample.unwrap_or(true) {
            true => {
                let range = number("clip_sample_range", hf.clip_sample_range, 1.0)?;
                Some(range as f32)
                    .filter(|limit| limit.is_finite() && *limit > 0.0)
                    .ok_or_else(|| {
                        invalid(format!(
                            "clip_sample_range: {range}, not a positive FP32 value"
                        ))
                    })?
            }
            false => f32::INFINITY,
        };
        Ok(Self {
            alpha,
            terminal,
            limit,
        })
    }
}

impl Config {
    pub fn timesteps(&self) -> usize {
        self.alpha.len()
    }

    pub fn alpha(&self) -> &[f32] {
        &self.alpha
    }

    pub fn terminal(&self) -> f32 {
        self.terminal
    }

    pub fn limit(&self) -> f32 {
        self.limit
    }
}

/// A DDIM scheduler's configuration: epsilon prediction without thresholding or zero-SNR
/// rescaling.
#[derive(Deserialize)]
pub struct Hf {
    num_train_timesteps: Option<usize>,
    prediction_type: Option<Literal<"epsilon">>,
    thresholding: Option<Disabled>,
    rescale_betas_zero_snr: Option<Disabled>,
    trained_betas: Option<Vec<f64>>,
    beta_start: Option<f64>,
    beta_end: Option<f64>,
    beta_schedule: Option<String>,
    set_alpha_to_one: Option<bool>,
    clip_sample: Option<bool>,
    clip_sample_range: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn parse(document: &Value) -> io::Result<Config> {
        super::super::parse::<Hf>(document).and_then(Config::try_from)
    }

    /// The cumulative alphas are those Diffusers 0.37.1's `DDIMScheduler` builds from trained
    /// betas, a linear and a scaled linear schedule, and the defaults it fills in.
    #[test]
    fn derives_the_alphas_diffusers_does() {
        // Diffusers rounds its schedule in FP32 steps where the parser rounds FP64 values once:
        // each product may add two FP32 units in the last place of difference.
        let close = |config: &Config, at: usize, want: f64| {
            let got = f64::from(config.alpha()[at]);
            let bound = (at + 1) as f64 * 2f64.powi(-22);
            assert!(
                (got - want).abs() <= want * bound,
                "{at}: {got}, not {want}"
            );
        };
        for (document, alphas) in [
            (
                json!({"num_train_timesteps": 2, "trained_betas": [0.1, 0.2]}),
                &[0.8999999761581421, 0.7199999690055847][..],
            ),
            (
                json!({"num_train_timesteps": 3, "beta_start": 0.1, "beta_end": 0.3}),
                &[0.8999999761581421, 0.7199999690055847, 0.5040000081062317],
            ),
            (
                json!({
                    "num_train_timesteps": 3, "beta_start": 0.01, "beta_end": 0.09,
                    "beta_schedule": "scaled_linear",
                }),
                &[0.9900000095367432, 0.9503999948501587, 0.864863932132721],
            ),
        ] {
            let config = parse(&document).unwrap();
            assert_eq!(config.timesteps(), alphas.len(), "{document}");
            for (at, &want) in alphas.iter().enumerate() {
                close(&config, at, want);
            }
        }
        let defaults = parse(&json!({})).unwrap();
        assert_eq!(
            (defaults.timesteps(), defaults.terminal(), defaults.limit()),
            (1000, 1.0, 1.0)
        );
        close(&defaults, 0, 0.9998999834060669);
        close(&defaults, 999, 4.035830352222547e-05);
        let unclipped = json!({"set_alpha_to_one": false, "clip_sample": false});
        let config = parse(&unclipped).unwrap();
        assert_eq!(
            (config.terminal(), config.limit()),
            (config.alpha()[0], f32::INFINITY)
        );
    }

    /// Each value the scheduler does not run is refused by kind and reason: another prediction,
    /// thresholding, rescaling or schedule as unsupported, a value no schedule holds as invalid.
    #[test]
    fn refuses_what_ddim_does_not_run() {
        use io::ErrorKind::{InvalidData, Unsupported};
        let infinite: Value = serde_json::from_str("1e9999").unwrap();
        let trained = json!({"num_train_timesteps": 2, "trained_betas": [0.1, 0.2]});
        let scheduled = json!({"num_train_timesteps": 2});
        let cases = [
            (
                &trained,
                json!({"prediction_type": "v_prediction"}),
                Unsupported,
                r#"prediction_type: found "v_prediction", compiled "epsilon""#.to_string(),
            ),
            (
                &trained,
                json!({"prediction_type": 3}),
                InvalidData,
                "prediction_type: invalid type: number, expected a string".into(),
            ),
            (
                &trained,
                json!({"thresholding": true}),
                Unsupported,
                "thresholding: found true, compiled false".into(),
            ),
            (
                &trained,
                json!({"rescale_betas_zero_snr": true}),
                Unsupported,
                "rescale_betas_zero_snr: found true, compiled false".into(),
            ),
            (
                &scheduled,
                json!({"num_train_timesteps": 0}),
                InvalidData,
                "num_train_timesteps: 0, a schedule has a timestep".into(),
            ),
            (
                &trained,
                json!({"trained_betas": [0.1]}),
                InvalidData,
                "trained_betas: 1 betas for 2 timesteps".into(),
            ),
            (
                &trained,
                json!({"trained_betas": [0.1, 1.0]}),
                InvalidData,
                "beta 1 at timestep 1, outside (0, 1)".into(),
            ),
            (
                &trained,
                json!({"trained_betas": [0.0, 0.2]}),
                InvalidData,
                "beta 0 at timestep 0, outside (0, 1)".into(),
            ),
            (
                &trained,
                json!({"trained_betas": [0.1, infinite]}),
                InvalidData,
                "beta inf at timestep 1, outside (0, 1)".into(),
            ),
            (
                &trained,
                json!({"num_train_timesteps": 8, "trained_betas": vec![0.999999; 8]}),
                InvalidData,
                "the alpha product underflows FP32 at timestep 7".into(),
            ),
            (
                &scheduled,
                json!({"beta_schedule": "squaredcos_cap_v2"}),
                Unsupported,
                r#"beta_schedule: "squaredcos_cap_v2", DDIM runs linear or scaled_linear"#.into(),
            ),
            (
                &scheduled,
                json!({"beta_end": infinite}),
                InvalidData,
                "beta_end: inf, not a finite number".into(),
            ),
            (
                &trained,
                json!({"clip_sample_range": 0}),
                InvalidData,
                "clip_sample_range: 0, not a positive FP32 value".into(),
            ),
            (
                &trained,
                json!({"clip_sample_range": 1e39}),
                InvalidData,
                format!(
                    "clip_sample_range: 1{}, not a positive FP32 value",
                    "0".repeat(39)
                ),
            ),
            (
                &trained,
                json!({"clip_sample_range": infinite}),
                InvalidData,
                "clip_sample_range: inf, not a finite number".into(),
            ),
        ];
        for (base, change, kind, reason) in cases {
            let mut document = base.clone();
            let fields = change.as_object().unwrap().clone();
            document.as_object_mut().unwrap().extend(fields);
            let error = parse(&document).err().unwrap();
            assert_eq!(
                (error.kind(), error.to_string()),
                (kind, reason),
                "{change}"
            );
        }
    }
}
