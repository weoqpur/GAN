## WGAN(Wasserstein GAN)

이 논문은 기존 GAN의 loss function에 주목하였다.

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

그럼 multi-modal은 무엇이냐 바로 mode가 여러개 존재한다는 말이다. mode dropping의
예를 들어보면 확률 분포표에 mode가 4개 있고 각각 자몽, 사과, 레몬, 오렌지라고 해보자 4개의 mode 중 하나로만 치우쳐서 변환시킬 때 문제가 생기며

하나의 mode로 치우칠 경우 generator가 계속 같은 이미지만 생성하고 discriminator는 그 이미지가 진짜 이미지라고 생각한다.
이렇게 계속 훈련이 되면 generator는 뭘 만들든 discriminator만 속이면 되기에 같은 이미지만 계속 만들고 
이미지의 품질은 올라갈 수 있어도 이미지의 다양성이 떨어져 문제가 된다.

