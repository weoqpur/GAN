dataset download link : 
<https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ>

img_align_celeba.zip 설치 후

압축을 풀어 "celeba" 라는 디렉토리를 생성 후 파일을 넣어준 뒤

경로를 root라는 변수에 문자열로 저장 해줍니다.

## DCGAN

### DCGAN이 나온 배경

DCGAN은 GAN모델에서 Generator를 개선한 논문이다.
일반 GAN은 Generator가 Discriminator를 속인다
이 하나만 목적으로 가지고 있어 생성된 이미지가 그 전에
생성한 이미지와 완전 다르거나 우리가 보았을때 무슨 이미지인지
유추도 안되는 상황이 발생하였다. 그래서 이 점을 보안하기 위해
나왔다.
GAN의 한계에서 자세히 설명하도록 하겠다.

### GAN의 한계

1. GAN은 결과가 불안정하다   
   이미지가 불규칙적이기에 성능이 좋다고할 수 없었다   
   

2. black-box method   
   GAN만 가지고 있는 단점은 아니지만, 결정 변수나 주요 변수를   
   알 수 있는 다수의 머신러닝 기법들과 달리 Neural Network는   
   처음부터 끝까지 결과에 대한 과정을 알 수 없다.
   

3. Generative Model평가   
   결과물이 얼마나 잘 만들어졌는지 비교 할 수 없다.   
   사람이 봐도 주관적인 기준이기 때문에 얼마나 뛰어난지, 정확한지   
   판단하기 어렵다.