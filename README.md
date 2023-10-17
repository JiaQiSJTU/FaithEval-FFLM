# Zero-shot Faithfulness Evaluation for Text Summarization with Foundation Language Model

This paper has been accepted by EMNLP2023.

## Requirements

* python==3.7
* pytorch==1.11.0
* transformers==4.28.1
* scipy==1.7.3
* scikit-learn==1.0.2
* numpy==1.21.5

## Prepare datasets

Download the benchmark datasets and put them under the directory ./data. 
Modify corresponding paths in load_dataset.py if necessary.

<table>
<thead>
  <tr>
    <th>Setting</th>
    <th>Dataset</th>
    <th>Val</th>
    <th>Test</th>
    <th>Source</th>
    <th>Link</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="6">Inconsistency Detection<br>(SUMMAC Benchmark)</td>
    <td>CoGenSum</td>
    <td>1281</td>
    <td>400</td>
    <td>C</td>
    <td rowspan="6">https://github.com/tingofurro/summac</td>
  </tr>
  <tr>
    <td>SummEval</td>
    <td>850</td>
    <td>850</td>
    <td>C</td>
  </tr>
  <tr>
    <td>FRANK</td>
    <td>671</td>
    <td>1575</td>
    <td>C+X</td>
  </tr>
  <tr>
    <td>Polytope</td>
    <td>634</td>
    <td>634</td>
    <td>C</td>
  </tr>
  <tr>
    <td>FactCC</td>
    <td>931</td>
    <td>503</td>
    <td>C</td>
  </tr>
  <tr>
    <td>XSumFaith</td>
    <td>1250</td>
    <td>1250</td>
    <td>C</td>
  </tr>
  <tr>
    <td rowspan="5">Faithfulness Rating</td>
    <td>FRANKCNN</td>
    <td>-</td>
    <td>1250</td>
    <td>C</td>
    <td rowspan="2">https://github.com/NJUNLP/CoP</td>
  </tr>
  <tr>
    <td>QAGSCNN</td>
    <td>-</td>
    <td>235</td>
    <td>C</td>
  </tr>
  <tr>
    <td>SummEval</td>
    <td>-</td>
    <td>1600</td>
    <td>C</td>
    <td>https://github.com/Yale-LILY/SummEval</td>
  </tr>
  <tr>
    <td>FRANKXSUM</td>
    <td>-</td>
    <td>996</td>
    <td>X</td>
    <td rowspan="2">https://github.com/NJUNLP/CoP</td>
  </tr>
  <tr>
    <td>QAGSXSUM</td>
    <td>-</td>
    <td>239</td>
    <td>X</td>
  </tr>
</tbody>
</table>

## Probability Caculation

Calculate the probabilities based on a foundation language model by:

```shell
CUDA_VISIBLE_DEVICES=0 python3 main.py
```

The results will be saved under the directory ./output, or can be downloaded with this [link](https://drive.google.com/drive/folders/1IhhmzXdjgndHTbKEweeaiN0VDcHRhMhE?usp=share_link).

## FFLM

Then, the summary-level and system-level performances of FFLM can be calculated as follows:

```shell
python3 summary-level-evaluation.py --file_path xxx
python3 system-level-evaluation.py --file_path xxx
```


## Citation

```
@article{jia2023fflm,
  title={Zero-shot Faithfulness Evaluation for Text Summarization with Foundation Language Model},
  author={Qi Jia, Siyu Ren, Yizhu Liu, Kenny Q. Zhu},
  jbooktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  year={2023}
}
```