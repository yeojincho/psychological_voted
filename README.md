# 프로젝트 목표
본 프로젝트는 개인의 심리 성향 데이터를 기반으로
투표 참여 여부(voted, 0~1)를 예측하는 머신러닝 모델을 구축하는 것을 목표로 한다.

특히 다음에 집중한다:

- 설문 응답(Q_A ~ Q_E 등) 기반의 심리적 특징 분석
- 유의미한 feature engineering을 통한 예측 성능 향상
- 다양한 모델(LightGBM, XGBoost 등)을 비교하여 최적 모델 도출

최종적으로는
👉 심리적 요인이 실제 행동(투표)에 미치는 영향을 데이터 기반으로 분석하고 예측하는 것을 목표로 한다.

---

# 데이터 설명

# 데이터 변수 설명

### **`Q_A (a~t)`**  질문

### **`Q_E (a~t)`** 답변에 걸린 시간

<aside>

- 1=동의하지 않음
- 2=약간 동의하지 않음
- 3=중립
- 4=약간 동의함
- 5=동의함
</aside>

```python
Qa : 비밀(비식별화를 위해)
Qb: 대부분의 범죄자와 일반인의 가장 큰 차이점은 범죄자들은 잡힐 만큼 어리석다는 것입니다.
Qc: 다른 사람을 완전히 신뢰하는 사람은 문제를 자초하는 겁니다.
질문: 비밀
질문: PT 바넘이 "세상엔 매분마다 속기 쉬운 사람이 한 명씩 태어난다"고 말한 건 틀린 말입니다.
Qf: 다른 사람에게 거짓말을 하는 것은 절대 용납될 수 없습니다.
Qg : 비밀
Qh: 대부분의 사람들은 재산 손실보다 부모님의 죽음을 더 쉽게 잊습니다.
치 : 비밀
Qj: 모든 사람에게는 악의적인 면이 있고, 기회가 주어지면 그 면이 드러날 것이라고 가정하는 것이 가장 안전합니다.
Qk: 결론적으로, 중요한 인물이 되려고 부정직하게 구는 것보다 겸손하고 정직한 것이 더 낫습니다.
Ql : 비밀
Qm: 요령을 부리지 않고서는 성공하기 어렵습니다.
질문: 비밀
질문: 사람들을 다루는 가장 좋은 방법은 그들이 듣고 싶어하는 말을 해주는 것이다.
Qp : 비밀
Qq: 대부분의 사람들은 기본적으로 선하고 친절합니다.
질문: 도덕적으로 옳다고 확신할 때만 행동해야 합니다.
질문: 중요한 사람들에게 아첨하는 것은 현명한 일이다.
Qt : 비밀
```

### `age_group` : 연령

<aside>

- 10s ~
</aside>

### `education` : 교육 수준

<aside>

- 1=고등학교 미만
- 2=고등학교 졸업
- 3=대학교 학위
- 4=대학원 학위
- 0=무응답
</aside>

### `engnat` : 모국어가 영어

<aside>

- 1=Yes
- 2=No
- 0=무응답
</aside>

### `familysize` : 형제자매 수

### `gender` : 성별

<aside>

- Female
- Male
</aside>

### `hand` : 필기하는 손

<aside>

- 1=Right
- 2=Left
- 3=Both
- 0=무응답
</aside>

### `married` : 혼인 상태

<aside>

- 1=Never married (미혼)
- 2=Currently married (기혼)
- 3=Previously married (과거 기혼)
- 0=Other (기타, 결측치 등)
</aside>

### `race` : 인종

<aside>

- Asian
- Arab
- Black
- Indigenous Australian
- Native American
- White
- Other
</aside>

### `religion` : 종교

<aside>

Agnostic, Atheist, Buddhist, Christian_Catholic, Christian_Mormon, Christian_Protestant, Christian_Other, Hindu, Jewish, Muslim, Sikh, Other

</aside>

### `tp__(01~07)` : 자신을 평가 ⇒ 답변 데이터 분포가 0~7

0~6(클수록 정도가 강함) 7은 무응답

<aside>

`문항`

- tp01: 외향적이고 열정적입니다.
- tp02 : 비판적이고, 다투기를 좋아하는.
- tp03: 믿음직스럽고 자기 절제력이 뛰어남.
- tp04: 불안해하고 쉽게 화를 낸다.
- tp05: 새로운 경험에 열려있고, 복잡한 성격을 지녔습니다.
- tp06 : 조용하고 한적한 곳.
- tp07 : 공감 능력이 뛰어나고 따뜻한.
- tp08: 정리정돈이 안 되어 있고, 부주의하다.
- tp09: 차분하고 감정적으로 안정적입니다.
- tp10: 틀에 박힌, 창의성이 부족한.
</aside>

<aside>

`답변`

- 1 = Disagree strongly
- 2 = Disagree moderately
- 3 = Disagree a little
- 4 = Neither agree nor disagree
- 5 = Agree a little
- 6 = Agree moderately
- 7 = Agree strongly
</aside>

### `urban` : 유년기의 거주 구역

<aside>

- 1=Rural (country side) 시골
- 2=Suburban 교외
- 3=Urban (town, city) 도시
- 0=무응답
</aside>

### `wr_(01~13)` : 실존하는 해당 단어의 정의을 앎

<aside>

- 1=예
- 0=아니오
</aside>

### `wf_(01~03)` : 허구인 단어의 정의를 앎

<aside>

- 1=예
- 0=아니오
</aside>

### **`voted` (타겟): 지난 해 국가 선거 투표 여부**

<aside>

- 1=예
- 2=아니오
</aside>
---

# 실행 방법
### 1. 환경 설정
```
pip install -r requirements.txt
```

### 2. 데이터 준비
- 데이콘에서 데이터 다운로드(https://dacon.io/competitions/official/236705/data)
- 아래 경로에 배치
```
data/ 
├── train.csv 
└── test.csv
```

---
### 참고
- [심리 성향 예측 AI 해커톤](https://dacon.io/competitions/official/236705/data)

- [데이터 변수 설명](https://www.dacon.io/competitions/official/235647/talkboard/401534?page=1&dtype=recent&ptype=pub)

- [참고 사이트](https://openpsychometrics.org/)