# UPSTAGE_DKT
# 1. Summary
### 학생의 학습을 예측하는 모델을 만들기 위해서 학생 개인의 학습 관련 데이터를 순차적으로 입력하여 다음 결과를 예측하는 모델이 필요하다. 본 실험에서는 DKVNM(Dyanamic Key Value Memory Network) 모델을 활용했으며, 이 모델은 학생의 학습기록을 메모리 네트워크에 저장해 시간에 따라 메모리가 업데이트 되는 구조를 가진 딥러닝모델이다. 또한 기존 DKVMN 모델에, Feature 데이터를 입력시키고 메모리 네트워크의 Attention 메커니즘에 Fuzzy Logic을 적용해 학습시키는 방법으로 모델을 개량하였다. 본 실험에서 제안하는 모델의 성능 비교를 위해 DKVMN와 AUC를 비교했으며 20Epochs 으로 학습하였다. 추후로 모델을 향상시키기 위해서는 메모리 네트워크를 추가 활용하는것이 좋을것으로 생각되며, Feature를 Bundle과 조합하여 활용하는 연구가 필요하다. 



# 2. Experimental results

* Epochs 20
|model|1|5|10|15|20|
|---|---|---|---|---|---|
|Vanillar DKVMN                           |0.6871|    0.7482|   0.7643|   0.7766|   0.7799|
|Fuzzy Logic DKVMN                        |0.6760|    0.7475|   0.7659|   0.7752|   0.7789|
|Feature AutoEncoder DKVMN                |0.5276|    0.5349|   0.5913|   0.6048|   0.6209|
|Feature AutoEncoder + Fuzzy Lgoic DKVMN  |0.5339|    0.6174|   0.6294|   0.6472|   0.6521|

* Epochs 50
|model|1|15|25|35|50|
|---|---|---|---|---|---|
|Vanillar DKVMN                            |0.6813|    0.7725|   0.7814|   0.7876|  0.7900|
|Fuzzy Logic DKVMN                         |0.6819|    0.7773|   0.7809|   0.7856|  0.7893|

* Epochs 300
|model|300|
|---|---|
|Vanillar DKVMN                            |0.8154|

# 3. Instruction
## (1) Prepare your data
#### - 데이터를 DKVMN에 학습 가능한 형태로 변환시키는 과정이 필요합니다.
```
Number of bundles    3
Bundle sequences     60, 60, 60
Correctness info     0, 1, 1
```
#### - 데이터를 위와 같이 변환하는 코드가 main.py에 포함되어있습니다. (default=False)
```bash
% python main.py --trans=True
```

## (2) Parsing & Run
#### - 모델에 추가할 Add-on을 설정합니다. 
#### 1) DKVMN (Default, Baseline)
```bash
% python main.py --auto_encoder=False --fuzzy_logic=False --feedforward=False

```
#### 2) DKVMN with Fuzzy_logic
```bash
% python main.py --auto_encoder=False --fuzzy_logic=True --feedforward=False

```
#### 3) DKVMN with Autoencoder
```bash
% python main.py --auto_encoder=True --fuzzy_logic=False --feedforward=False

```
#### 4) DKVMN with LSTM (as feedforward)
```bash
% python main.py --auto_encoder=False --fuzzy_logic=False --feedforward=True

```
## (3) OUTPUT
#### 총 세가지 파일이 결과물로 나옵니다. 
~~~
model.pth
train_result.pickle
pred.csv
~~~

# 4. Approach
## (1) Feature
### 기존 BKT(Yudelson et al., 2013)<sup>[10](#footnote_10)</sup>, DKT(Piech et al., 2017)<sup>[4](#footnote_4)</sup>, DKVMN(Zhang et al., 2017)<sup>[11](#footnote_11)</sup>와 같은 Knowledg Tracing의 모델은 학생들의 학습기록의 결과(correctness) 정보 만으로 네트워크를 학습시킨다. 이러한 모델을 응용하여 이후의 여러 연구에서 Feature를 추가한 모델을 발표하였다. Zhang et al. (2017)<sup>[12](#footnote_12)</sup>은 학습관련 Feature를 오토인코더를 활용하여 차원축소 이후 LSTM 기반의 DKT모델에 적용하여 성능을 향상시켰다. Sun et al. (2021)<sup>[7](#footnote_7)</sup>의 모델에서는 학습능력 Feature를 KMeans 알고리즘을 활용해 학생들을 세 그룹으로 나누었으며, 이를 DKVNM의 메모리 업데이트 부분과 예측부분에 추가하여, 기존의 모델에 Feature를 추가로 학습시키는 방법론을 제시했다. <br>
### 현재의 태스크에서는 기존의 연구를 바탕으로 오토인코더를 적용해 차원 축소한 후에 $p$(t)를 예측하는 방법으로 모델을 구성했다. 실험결과 Feature를 직접적으로 예측 모델에 사용할 경우 AUC(Area under the ROC Curve))가 낮은것으로 확인하였으며 메모리 네트워크에 구성되는 방향으로 활용하는것이 더 좋을것으로 보인다. 이는 Feature 데이터의 의미를 학생들 개인으로부터 측정된 데이터로 추론할 수 있으며, 이 데이터를 카테고리 범주형 데이터로 활용될 수 있다는 점에서 메모리 네트워크에 임베딩 된 Bundle 데이터와 결합하여 모델을 발전시키는 방향을 제안한다.<br>
## (2) Attention 
### DKVMN에는 Attention(Vaswani et al., 2017)<sup>[9](#footnote_9)</sup> 구조로 메모리 네트워크를 구성한다. 자연어처리에서는 Attention 메커니즘을 응용하여 여러 논문들이 발표되고 있으며(Singh, 2021)<sup>[6](#footnote_6)</sup>, Knowledge Tracing 태스크에서 Abdelrahman & Wang, (2019)<sup>[1](#footnote_1)</sup> 의 모델은 어텐션 벡터를 통해 문제간(Exercises)의 연관성을 Fuzzy logic으로 추출한 후에 LSTM 네트워크를 통해 결과를 예측하는 모델을 발표했다.<br>
### 사전 연구를 기반으로 각 exercise의 개념이 비슷할수록 가중치를 부여하도록 Triangular Membership(Pedrycz, 1994)<sup>[3](#footnote_3)</sup>을 활용하였다. 구체적으로는, Attention 벡터내에서 KMeans로 세 개의 중심점을 구한 뒤 Exercise의 Attention weight에 Logic을 적용해, 관련된 문제 일수록 비슷한 값을 가지도록 모델을 구성하였다. Fuzzy logic을 활용한 모델의 경우 Baseline모델보다 AUC 값이 약간 향상되어있는것을 확인 하였으며, 오토인코더를 활용한 모델에서 Fuzzy logic을 활용할 경우 AUC가 더 높은것으로보아 추후 연구에서 어텐션 벡터를 활용하여 더 높은 결과를 만들 수 있을것으로 기대된다.<br>
## (3) Domain Knowledge
### DKVMN과 같은 메모리 네트워크(Santoro, 2016)<sup>[5](#footnote_5)</sup> 의 특징은 정보를 연결하고 메모리 내의 구조적 관계를 통해 예측 이나 분류 모델의 성능을 발전시킨다는 점이다. 이는 사람의 기억하는 방법을 딥러닝으로 구조화한것으로 해석가능하다. 이와 대응하여, 학습관련 모델을 개발하는데 있어 교육학에 관련된 도메인 지식을 활용하여 모델을 향상시키는 방법을 고려할 수 있다. Kim et al. (2021)<sup>[2](#footnote_2)</sup> 의 논문에서는 틀린문제와 맞춘문제에 대한 정보를 인코딩 하여 학습자의 Insight를 모델 내부에 구성했다. <br>
### 본 연구에서는 다루지 않았지만, 교육학적인 관점 아래에서 학생들이 성적을 맞추고 난 이후의 결과값을 추출하여 심리적인 영향을 줄수있음을 모델에 추가하는것을 제안하고자한다. Uguroglu & Walberg (1979)<sup>[8](#footnote_8)</sup>은 동기부여이론 에서는 정량적인 결과가 학습된 무기력 혹은 자기효능감과 같은 학습 동기에 요인이 될 수 있다 제시한다. 따라서 학생의 학습 결과를 학생의 동기부여로 활용하여 모델에 추가적인 기능을 생성할 수 있다. 예를들어 시험점수의 평균과 같은 데이터를 동기부여 요소로 이용하여, 시계열 분석 방법론을 적용하면 학생의 동기 정보를 학습 가능할것이다.<br>

##### Closing Remarks 
##### 본 프로젝트를 진행하며 컴퓨터성능이 더 좋았으면 이라는 생각을 많이 했습니다. Epochs를 300으로 할당한 실험이 36시간 가량 걸렸기 때문에 여러 모델을 비교해서 학습하려는 초기 계획이 틀어진것 같아 많아 아쉽습니다. 또한 LSTM으로 학습하면 현재 장비로는 메모리가 부족해 자동 종료되는 상황에 당황했던 기억이 납니다. :` 더군다나 더 좋은 결과를 위해 제출 시간을 늘려주심에 너무나 감사드립니다. 채용에 관계없이 업스테이지 인사팀의 친절에 감동먹습니다. <br> 부디 다음 테스트를 보시는 분들은 더 많은 기능을 추가한 모델을 더 좋은 환경에서 사용하기를 기원드립니다.

##### References
<a name="footnote_1">[1]</a> Abdelrahman, G., & Wang, Q. (2019, July). Knowledge tracing with sequential key-value memory networks. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 175-184).<br>
<a name="footnote_2">[2]</a> Kim, S., Kim, W., Jung, H., & Kim, H. (2021, June). DiKT: Dichotomous Knowledge Tracing. In International Conference on Intelligent Tutoring Systems (pp. 41-51). Springer, Cham.
<a name="footnote_3">[3]</a> Pedrycz, W. (1994). Why triangular membership functions?. Fuzzy sets and Systems, 64(1), 21-30.
<a name="footnote_4">[4]</a> Piech, C., Spencer, J., Huang, J., Ganguli, S., Sahami, M., Guibas, L., & Sohl-Dickstein, J. (2015). Deep knowledge tracing. arXiv preprint arXiv:1506.05908.<br>
<a name="footnote_5">[5]</a> Santoro, A., Bartunov, S., Botvinick, M., Wierstra, D., & Lillicrap, T. (2016, June). Meta-learning with memory-augmented neural networks. In International conference on machine learning (pp. 1842-1850). PMLR.<br>
<a name="footnote_6">[6]</a> Singh, L. (2021, July). Clustering Text Using Attention. In 2021 12th International Conference on Computing Communication and Networking Technologies (ICCCNT) (pp. 1-5). IEEE.<br>
<a name="footnote_7">[7]</a> Sun, X., Zhao, X., Li, B., Ma, Y., Sutcliffe, R., & Feng, J. (2021). Dynamic Key-Value Memory Networks With Rich Features for Knowledge Tracing. IEEE Transactions on Cybernetics. <br>
<a name="footnote_8">[8]</a> Uguroglu, M. E., & Walberg, H. J. (1979). Motivation and achievement: A quantitative synthesis. American educational research journal, 16(4), 375-389.<br>
<a name="footnote_9">[9]</a> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).<br>
<a name="footnote_10">[10]</a> Yudelson, M. V., Koedinger, K. R., & Gordon, G. J. (2013, July). Individualized bayesian knowledge tracing models. In International conference on artificial intelligence in education (pp. 171-180). Springer, Berlin, Heidelberg.<br>
<a name="footnote_11">[11]</a> Zhang, J., Shi, X., King, I., & Yeung, D. Y. (2017, April). Dynamic key-value memory networks for knowledge tracing. In Proceedings of the 26th international conference on World Wide Web (pp. 765-774).<br>
<a name="footnote_12">[12]</a> Zhang, L., Xiong, X., Zhao, S., Botelho, A., & Heffernan, N. T. (2017, April). Incorporating rich features into deep knowledge tracing. In Proceedings of the fourth (2017) ACM conference on learning@ scale (pp. 169-172. <br>





