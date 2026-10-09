# Phase 2.9 Weighted vs Unweighted Comparison

Selection uses the frozen validation split only. The test set is not used.

## logistic_regression

| Metric | Unweighted | Weighted | Difference |
| --- | ---: | ---: | ---: |
| accuracy | 0.846315 | 0.840485 | -0.005830 |
| macro_precision | 0.873004 | 0.848833 | -0.024172 |
| macro_recall | 0.826383 | 0.858851 | +0.032469 |
| macro_f1 | 0.837739 | 0.852986 | +0.015247 |
| weighted_f1 | 0.835213 | 0.842121 | +0.006908 |

| Class | Metric | Unweighted | Weighted | Difference |
| --- | --- | ---: | ---: | ---: |
| Change | precision | 0.987893 | 0.948661 | -0.039233 |
| Change | recall | 0.931507 | 0.970320 | +0.038813 |
| Change | f1 | 0.958872 | 0.959368 | +0.000496 |
| Incident | precision | 0.759445 | 0.833021 | +0.073575 |
| Incident | recall | 0.923256 | 0.774419 | -0.148837 |
| Incident | f1 | 0.833377 | 0.802651 | -0.030726 |
| Problem | precision | 0.763941 | 0.625864 | -0.138077 |
| Problem | recall | 0.455654 | 0.702882 | +0.247228 |
| Problem | f1 | 0.570833 | 0.662141 | +0.091308 |
| Request | precision | 0.980738 | 0.987785 | +0.007047 |
| Request | recall | 0.995114 | 0.987785 | -0.007329 |
| Request | f1 | 0.987874 | 0.987785 | -0.000089 |

Largest changed off-diagonal confusion counts (weighted minus unweighted):
- Incident → Problem: +250
- Problem → Incident: -227
- Change → Request: -9
- Change → Incident: -8
- Request → Change: +8
- Incident → Change: +7
- Problem → Change: +3
- Request → Problem: +2
- Incident → Request: -1
- Problem → Request: +1

## linear_svm

| Metric | Unweighted | Weighted | Difference |
| --- | ---: | ---: | ---: |
| accuracy | 0.874067 | 0.876632 | +0.002565 |
| macro_precision | 0.888666 | 0.886735 | -0.001932 |
| macro_recall | 0.867811 | 0.879182 | +0.011371 |
| macro_f1 | 0.875505 | 0.882558 | +0.007053 |
| weighted_f1 | 0.870374 | 0.875332 | +0.004958 |

| Class | Metric | Unweighted | Weighted | Difference |
| --- | --- | ---: | ---: | ---: |
| Change | precision | 0.983645 | 0.983834 | +0.000189 |
| Change | recall | 0.961187 | 0.972603 | +0.011416 |
| Change | f1 | 0.972286 | 0.978186 | +0.005900 |
| Incident | precision | 0.816176 | 0.839106 | +0.022930 |
| Incident | recall | 0.903488 | 0.873256 | -0.030233 |
| Incident | f1 | 0.857616 | 0.855840 | -0.001775 |
| Problem | precision | 0.766990 | 0.736145 | -0.030846 |
| Problem | recall | 0.613082 | 0.677384 | +0.064302 |
| Problem | f1 | 0.681454 | 0.705543 | +0.024089 |
| Request | precision | 0.987854 | 0.987854 | +0.000000 |
| Request | recall | 0.993485 | 0.993485 | +0.000000 |
| Request | f1 | 0.990662 | 0.990662 | +0.000000 |

Largest changed off-diagonal confusion counts (weighted minus unweighted):
- Problem → Incident: -58
- Incident → Problem: +51
- Change → Incident: -4
- Change → Request: -1
- Incident → Request: +1

## distilbert

| Metric | Unweighted | Weighted | Difference |
| --- | ---: | ---: | ---: |
| accuracy | 0.836054 | 0.822062 | -0.013993 |
| macro_precision | 0.846318 | 0.834573 | -0.011745 |
| macro_recall | 0.838911 | 0.845833 | +0.006923 |
| macro_f1 | 0.841500 | 0.838771 | -0.002730 |
| weighted_f1 | 0.832984 | 0.824635 | -0.008349 |

| Class | Metric | Unweighted | Weighted | Difference |
| --- | --- | ---: | ---: | ---: |
| Change | precision | 0.965986 | 0.951220 | -0.014767 |
| Change | recall | 0.972603 | 0.979452 | +0.006849 |
| Change | f1 | 0.969283 | 0.965129 | -0.004154 |
| Incident | precision | 0.781504 | 0.814459 | +0.032956 |
| Incident | recall | 0.840116 | 0.740116 | -0.100000 |
| Incident | f1 | 0.809751 | 0.775510 | -0.034240 |
| Problem | precision | 0.644330 | 0.578348 | -0.065982 |
| Problem | recall | 0.554324 | 0.675166 | +0.120843 |
| Problem | f1 | 0.595948 | 0.623018 | +0.027070 |
| Request | precision | 0.993453 | 0.994267 | +0.000814 |
| Request | recall | 0.988599 | 0.988599 | +0.000000 |
| Request | f1 | 0.991020 | 0.991425 | +0.000405 |

Largest changed off-diagonal confusion counts (weighted minus unweighted):
- Incident → Problem: +167
- Problem → Incident: -112
- Incident → Change: +5
- Change → Request: -2
- Problem → Change: +2
- Change → Incident: -1
- Problem → Request: +1
- Request → Incident: -1
- Request → Problem: +1

## Current policy

```json
{
  "distilbert": {
    "dataset_version": "ds_6fa2f19cf683",
    "decision_rationale": "The unweighted model achieved higher validation Macro F1, accuracy, and weighted F1. Balanced weighting improved macro recall slightly, but not the primary selection metric. This decision is specific to the current data, configuration, and validation objective; it does not establish that class weighting is universally ineffective.",
    "experiment_configuration_references": {
      "unweighted": "artifacts/classification/run/transformer_training/training_config.json",
      "weighted": "artifacts/classification/run/class_weighting/transformer/training_config.json"
    },
    "experiment_manifest_references": {
      "unweighted": "artifacts/classification/run/transformer_training/training_manifest.json",
      "weighted": "artifacts/classification/run/class_weighting/transformer/training_manifest.json"
    },
    "final_test_evaluation_status": "PENDING",
    "known_limitations": [
      "Validation evidence only",
      "Loss values across weighting policies are not directly comparable"
    ],
    "label_order": [
      "Change",
      "Incident",
      "Problem",
      "Request"
    ],
    "model_family": "transformer",
    "model_name": "distilbert/distilbert-base-uncased",
    "per_class_changes": {
      "Change": {
        "f1": {
          "difference": -0.004153917620365655,
          "unweighted": 0.9692832764505119,
          "weighted": 0.9651293588301463
        },
        "precision": {
          "difference": -0.014766882362701184,
          "unweighted": 0.9659863945578231,
          "weighted": 0.9512195121951219
        },
        "recall": {
          "difference": 0.006849315068493178,
          "unweighted": 0.9726027397260274,
          "weighted": 0.9794520547945206
        }
      },
      "Incident": {
        "f1": {
          "difference": -0.03424042634705893,
          "unweighted": 0.8097506304286916,
          "weighted": 0.7755102040816326
        },
        "precision": {
          "difference": 0.03295585758690256,
          "unweighted": 0.7815035154137372,
          "weighted": 0.8144593730006398
        },
        "recall": {
          "difference": -0.10000000000000009,
          "unweighted": 0.8401162790697675,
          "weighted": 0.7401162790697674
        }
      },
      "Problem": {
        "f1": {
          "difference": 0.027070346198281414,
          "unweighted": 0.5959475566150179,
          "weighted": 0.6230179028132993
        },
        "precision": {
          "difference": -0.06598231855963821,
          "unweighted": 0.6443298969072165,
          "weighted": 0.5783475783475783
        },
        "recall": {
          "difference": 0.12084257206208426,
          "unweighted": 0.5543237250554324,
          "weighted": 0.6751662971175166
        }
      },
      "Request": {
        "f1": {
          "difference": 0.00040466329447252924,
          "unweighted": 0.9910204081632653,
          "weighted": 0.9914250714577378
        },
        "precision": {
          "difference": 0.0008136391115114439,
          "unweighted": 0.9934533551554828,
          "weighted": 0.9942669942669943
        },
        "recall": {
          "difference": 0.0,
          "unweighted": 0.988599348534202,
          "weighted": 0.988599348534202
        }
      }
    },
    "reasoning": "Balanced weighting did not improve validation Macro F1; retain the simpler unweighted policy.",
    "selected_checkpoint_reference": "artifacts/classification/run/transformer_training/checkpoints/checkpoint-3753",
    "selected_experiment": "unweighted",
    "selected_validation_macro_f1": 0.8415004679143716,
    "selected_weighting_policy": "none",
    "selection_metric": "validation_macro_f1",
    "split_version": "split_d1f67d41500f",
    "status": "SELECTED",
    "transformer_dataset_version": "transformer_ds_340aff24d262",
    "unweighted_validation_macro_f1": 0.8415004679143716,
    "validation_macro_f1": 0.8415004679143716,
    "validation_macro_f1_difference": -0.002729833618667632,
    "weighted_validation_macro_f1": 0.838770634295704
  },
  "linear_svm": {
    "known_limitations": [
      "Validation evidence only",
      "Loss values across weighting policies are not directly comparable"
    ],
    "per_class_changes": {
      "Change": {
        "f1": {
          "difference": 0.005899618977417087,
          "unweighted": 0.9722863741339491,
          "weighted": 0.9781859931113662
        },
        "precision": {
          "difference": 0.00018885843171956296,
          "unweighted": 0.9836448598130841,
          "weighted": 0.9838337182448037
        },
        "recall": {
          "difference": 0.011415525114155223,
          "unweighted": 0.9611872146118722,
          "weighted": 0.9726027397260274
        }
      },
      "Incident": {
        "f1": {
          "difference": -0.0017754381992792778,
          "unweighted": 0.8576158940397351,
          "weighted": 0.8558404558404559
        },
        "precision": {
          "difference": 0.022929674663161315,
          "unweighted": 0.8161764705882353,
          "weighted": 0.8391061452513966
        },
        "recall": {
          "difference": -0.03023255813953485,
          "unweighted": 0.9034883720930232,
          "weighted": 0.8732558139534884
        }
      },
      "Problem": {
        "f1": {
          "difference": 0.024088627822624775,
          "unweighted": 0.6814540973505854,
          "weighted": 0.7055427251732102
        },
        "precision": {
          "difference": -0.030845712948882942,
          "unweighted": 0.7669902912621359,
          "weighted": 0.736144578313253
        },
        "recall": {
          "difference": 0.06430155210643018,
          "unweighted": 0.6130820399113082,
          "weighted": 0.6773835920177383
        }
      },
      "Request": {
        "f1": {
          "difference": 0.0,
          "unweighted": 0.9906617945594803,
          "weighted": 0.9906617945594803
        },
        "precision": {
          "difference": 0.0,
          "unweighted": 0.9878542510121457,
          "weighted": 0.9878542510121457
        },
        "recall": {
          "difference": 0.0,
          "unweighted": 0.993485342019544,
          "weighted": 0.993485342019544
        }
      }
    },
    "reasoning": "Balanced weighting improved validation Macro F1; inspect per-class deltas before final use.",
    "selected_weighting_policy": "balanced",
    "status": "SELECTED",
    "validation_macro_f1": 0.8825577421711281,
    "validation_macro_f1_difference": 0.007053202150190674
  },
  "logistic_regression": {
    "known_limitations": [
      "Validation evidence only",
      "Loss values across weighting policies are not directly comparable"
    ],
    "per_class_changes": {
      "Change": {
        "f1": {
          "difference": 0.0004960304302732421,
          "unweighted": 0.9588719153936546,
          "weighted": 0.9593679458239278
        },
        "precision": {
          "difference": -0.03923274818401934,
          "unweighted": 0.9878934624697336,
          "weighted": 0.9486607142857143
        },
        "recall": {
          "difference": 0.038812785388127935,
          "unweighted": 0.9315068493150684,
          "weighted": 0.9703196347031964
        }
      },
      "Incident": {
        "f1": {
          "difference": -0.030725665362370203,
          "unweighted": 0.8333770663867751,
          "weighted": 0.8026514010244049
        },
        "precision": {
          "difference": 0.07357539638744803,
          "unweighted": 0.7594452415112386,
          "weighted": 0.8330206378986866
        },
        "recall": {
          "difference": -0.14883720930232558,
          "unweighted": 0.9232558139534883,
          "weighted": 0.7744186046511627
        }
      },
      "Problem": {
        "f1": {
          "difference": 0.09130765883376857,
          "unweighted": 0.5708333333333333,
          "weighted": 0.6621409921671019
        },
        "precision": {
          "difference": -0.13807674946880144,
          "unweighted": 0.7639405204460966,
          "weighted": 0.6258637709772952
        },
        "recall": {
          "difference": 0.24722838137472286,
          "unweighted": 0.4556541019955654,
          "weighted": 0.7028824833702882
        }
      },
      "Request": {
        "f1": {
          "difference": -8.887215312869401e-05,
          "unweighted": 0.9878738884397736,
          "weighted": 0.987785016286645
        },
        "precision": {
          "difference": 0.0070466535258102425,
          "unweighted": 0.9807383627608347,
          "weighted": 0.987785016286645
        },
        "recall": {
          "difference": -0.007328990228013033,
          "unweighted": 0.995114006514658,
          "weighted": 0.987785016286645
        }
      }
    },
    "reasoning": "Balanced weighting improved validation Macro F1; inspect per-class deltas before final use.",
    "selected_weighting_policy": "balanced",
    "status": "SELECTED",
    "validation_macro_f1": 0.8529863388255199,
    "validation_macro_f1_difference": 0.0152472879371357
  }
}
```

## Transformer decision

Selected weighting policy: **none**.

Selected experiment: **unweighted**.

Selected checkpoint: `artifacts/classification/run/transformer_training/checkpoints/checkpoint-3753`.

The unweighted model achieved higher validation Macro F1, accuracy, and weighted F1. Balanced weighting improved macro recall slightly, but not the primary selection metric. This decision is specific to the current data, configuration, and validation objective; it does not establish that class weighting is universally ineffective.

The decision uses validation metrics only. Weighted and unweighted losses optimize different objectives and are not compared as evidence of model improvement.

Final test evaluation remains pending and is not part of weighting-policy selection.
