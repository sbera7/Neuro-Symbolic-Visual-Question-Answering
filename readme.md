# Neuro-Symbolic AI for Visual Question Answering
## Sort-of-CLEVR Dataset

Neuro-Symbolic AI allows us to combine Deep Learning’s superior pattern recognition abilities with the reasoning abilities of symbolic methods like program synthesis. This repository is an implementation of NSAI for Visual Question Answering on the Sort-of-CLEVR dataset using PyTorch. This implementation is inspired by the [Neuro-Symbolic VQA](https://arxiv.org/abs/1810.02338) paper by MIT-IBM Watson AI Lab.

The basic idea behind using NSAI for VQA is parsing the visual scene into a symbolic representation and using NLP to parse the query into an executable program which the program executor can use on the scene to find the answer. This implementation gets more than 99% on the Sort-of-CLEVR dataset.

## Requirements
- Pytorch <= 1.7
- Torchtext <= 0.8.0
- Torchvision <= 0.8.0
- OpenCV
- dlib
- Scikit Learn
- Pandas
- Numpy

## Usage
### Generate the dataset
```bash
# Generating 50 images for training and 100 images for testing
python data_generator.py --n_train 50 --n_test 100
```

## References
- [Neural-Symbolic VQA: Disentangling Reasoning from Vision and Language Understanding](https://arxiv.org/abs/1810.02338)
- [The Neuro-Symbolic Concept Learner: Interpreting Scenes, Words, and Sentences From Natural Supervision](https://arxiv.org/abs/1904.12584)
- [Seq2Seq Transformer](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/seq2seq_transformer/seq2seq_transformer.py)
- [Object detection using dlib](https://www.learnopencv.com/training-a-custom-object-detector-with-dlib-making-gesture-controlled-applications/)
