## WGAN(Wasserstein GAN)

WGAN은 기존 GAN이 학습이 잘 안되는 이유를 KL divergence의 한계로 언급하며, loss function을 재정의하여 안정성을 제고한다.
여기서 Wasserstein distance에 대해서 자세히 알아보도록 하겠다.

이 논문은 기존 GAN의 loss function에 주목하였다.
![`이미지`](/Users/sinmugyeol/Desktop/WGANNotation.png)
수식이 상당히 많이 나오는데 위 표기법을 참고하겠다.



![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F4HgQb%2Fbtqu2IouBYN%2FdCQNXSAl4MS8F8ZkAKFkjk%2Fimg.png)   
위 식에서는 P(x)의 식을 직접 표현하는 것은 어렵기 때문에(정답을 이미 알고있다는 의미가 된다),
GAN에서는 x를 결정하는 latent variable z의 분포를 가정하여 입력으로 대입하고, discriminator와
generator간의 관계를 학습시킴으로써 generator의 분포를 P(x)에 가깝게 학습시키고자 한다.

그러나 GAN의 문제점은 discriminator와 generator간의 균형을 유지하며 학습하기 어렵고, 학습이 완료된 후에도 mode dropping이 발생한다는 것이다.
이러한 문제가 발생하는 이유는, discriminator가 선생님 역할을 충분히 하지 못해 모델이 최적점까지 학습되지 못했기 때문이다.

### Mode Dropping

GAN의 학습 과정에서 특정 학습 이터레이션에서 discriminator의 학습과 generator의 학습이 서로를 상쇄하는 문제가 생길 수 있다.   
![`이미지`](https://1.bp.blogspot.com/-vgiN_5VQAM8/WZkIIcklDOI/AAAAAAAAAJo/cVjRqFYVUqIQiCW7fa4sOxqlt1eLaxyMwCEwYBhgL/s1600/12.png)   
이처럼 discriminator와 generator가 서로를 속고 속이며 제자리를 맴돈다면 양쪽 모두 전역해로 수렴할 수 없게 된다.

위에 문제점이 얽히고 설켜서 나타나는 대표적인 현상이 mode dropping이다. 특이나 학습데이터의 분포가 multi-modal 한 경우에 그러한 현상이 두드러질 수 있다.
실제로 많은 데이터가 multi-modal 이기에 문제가 되는 것이다.

Mode Dropping이 무엇인지 이해를 하려면 mode가 무엇인지 부터 이해해야 한다. mode는 최빈값, 즉 가장 빈도가 높은 값을 말합니다.
Mode는 확률 분포에서 가장 점이 많은 지점이라고 할 수 있다.

그럼 multi-modal은 무엇이냐 바로 mode가 여러개 존재한다는 말이다. mode droppingㅉ
예를 들어보면 확률 분포표에 mode가 4개 있고 각각 자몽, 사과, 레몬, 오렌지라고 해보자 4개의 mode 중 하나로만 치우쳐서 변환시킬 때 문제가 생기며

하나의 mode로 치우칠 경우 generator가 계속 같은 이미지만 생성하고 discriminator는 그 이미지가 진짜 이미지라고 생각한다.
이렇게 계속 훈련이 되면 generator는 뭘 만들든 discriminator만 속이면 되기에 같은 이미지만 계속 만들고 
이미지의 품질은 올라갈 수 있어도 이미지의 다양성이 떨어져 문제가 된다.

---

Wasserstein GAN에서는 이러한 문제점을 해결하기 위하여 기존의 GAN에 비해 아래와 같은 차이점을 둔다.   

+ discriminator대신 새로 정의한 critic을 사용한다. discriminator는 가짜/진짜를 판별하기 위해 sigmoid를 사용하고,
output은 가짜/진짜에 대한 예측 확률 값이다.
  
+ 반면 critic은 EM(Earth Mover) distance로부터 얻은 scalar 값을 이용한다.
+ EM(Earth Mover) distance는 확률 분포 간의 거리를 측정하는 척도 중 하나인데, 그 동안 일반적으로 사용된 척도는 KL divergence이다.
KL divergence는 매우 strict 하게 거리를 측정하는 방법이라서, continuous 하지 않은 경우가 있고 학습시키기 어렵다.
  
결과적으로, GAN의 discriminator보다 선생님 역할을 잘 수행할 수 있는 critic을 사용함으로써 gradient를 잘 전달시키고,
critic과 generator를 최적점까지 학습할 수 있다는 것이다.

+ training 시 discriminator와 generator간의 balance를 주의깊게 살피고 있지 않아도 된다.
+ GAN에서 일반적으로 발생되는 문제인 mode dropping을 해결 가능하다.

## Different Distances

### 1. Total Variation (TV) distance

![`이미지`](/Users/sinmugyeol/Desktop/TotalVariationDistance.png)   
두 확률 분포의 측정값이 벌어질 수 있는 가장 큰 값을 뜻한다.

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FOIjhB%2Fbtqu0uTsP6L%2FqWKxG6F0W0MKB7sA3liKpK%2Fimg.png)   
위 그림을 보면 쉽게 이해할 수 있다. 그림에서 빨간색 A의 영역 안에 있는 A들을 대입하였을 때, Pr(A)와 Pg(A)의 값의 차 중 가장 큰 것을 뜻한다.

### 2. Kullback-Leibler(KL) divergence

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FztpKH%2FbtquL0dRXvF%2FhPeB8bQU57lavQh1uydXA0%2Fimg.png)   
두 확률 분포의 차이를 계산하는데 사용하는 함수이다. 어떤 이상적인 분포에 대해 그 분포를 근사하는 다른 분포를 사용해 샘플링을 한다면 발생할 수 있는 정보 엔트로피 차이를 계산한다. 또한, 상대 엔트로피, 정보 획득량, 인포메이션 다이버전스라고도 한다.

이렇게 들으면 이해가 잘 안간다. 쉽게 설명하자면 그냥 Cross Entropy - Entropy라고 생각하면 된다. 즉 다시말해, KL divergence는 p와 q의 cross entropy에서 p의 entropy를 뺀 것이다. 결과적으로 두 분포의 차이를 나타낸다. 이것을 식으로 표현하면 이렇게할 수 있다.   
![`이미지`](https://wikimedia.org/api/rest_v1/media/math/render/svg/571a31aedae1107f2574a3dfcb4d6caf4924dcc5)   
KL divergence의 정확한 식은 이렇다. 우리가 대개 cross entropy를 minimize 하는 것은, 어차피 H(p)는 고정된 상수값이기 때문에 결과적으로는 KL-divergence를 minimize하는 것과 같다.

### 4.Earth-Mover (EM) distance

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcRYWX7%2FbtquPLFEIno%2Fb8ZCkbO7JObbR2XKzgeDt0%2Fimg.png)   
 
솔직이 나도 이 수식은 아직 이해가 안 간다. 그림으로 표현된 쉬운 설명으로 알아보겠다.
두 확률 분포의 결합확률분포 Π(Pr, Pg)중에서 d(X,Y) (x와 y의 거리)의 기댓값을 가장 작게 추정한 값이다.   

![`이미지`](https://blog.kakaocdn.net/dn/bj3bPu/btqu0tAfLOo/KK8IrApFXoTXzowJ9fZcYK/img.png)

위 그림을 보면 각각 파란색 원이 X의 분포, 빨간색 원이 Y의 분포, x가 결합 확률 분포를 의미한다. 그리고 초록색 선의 길이가 ||x-y||를 의미하는데, 즉 초록색 선 길이들의 기댓값을 가장 작게 추정한 값이다.


