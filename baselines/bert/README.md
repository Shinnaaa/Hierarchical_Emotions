# GoEmotions Pytorch

This is a Pytorch Implementation of [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) with [Huggingface Transformers](https://github.com/huggingface/transformers) 

We added some new matrix in it, to make a in-depth comparison with our method.

## What is GoEmotions

Dataset labeled **58000 Reddit comments** with **28 emotions**

- admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise + neutral

### Requirements

- torch==1.4.0
- transformers==2.11.0
- attrdict==2.0.1

### Hyperparameters

You can change the parameters from the json files in `config` directory.

| Parameter         |      |
| ----------------- | ---: |
| Learning rate     | 5e-5 |
| Warmup proportion |  0.1 |
| Epochs            |   10 |
| Max Seq Length    |   50 |
| Batch size        |   16 |

## How to Run

```bash
$ python3 run_goemotions.py --taxonomy original
```

## Reference

- [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
- [GoEmotions Github](https://github.com/google-research/google-research/tree/master/goemotions)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
