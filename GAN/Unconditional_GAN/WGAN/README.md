## WGAN(Wasserstein GAN)

이 논문은 기존 GAN의 loss function에 주목하였다.

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F4HgQb%2Fbtqu2IouBYN%2FdCQNXSAl4MS8F8ZkAKFkjk%2Fimg.png)   
위 식에서는 P(x)의 식을 직접 표현하는 것은 어렵기 때문에(정답을 이미 알고있다는 의미가 된다),
GAN에서는 x를 결정하는 latent variable z의 분포를 가정하여 입력으로 대입하고, discriminator와
generator간의 관계를 학습시킴으로써 generator의 분포를 P(x)에 가깝게 학습시키고자 한다.

그러나
