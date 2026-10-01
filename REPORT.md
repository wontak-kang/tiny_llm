# A / B / C 모델 비교

공통 조건: temperature 0.8, top_k 40, 130토큰 생성, **시드 7 고정** (모델 차이만 보이도록 난수를 맞춤)

| | 파라미터 | 구조 | 학습 토큰 | 학습 시간 | val loss |
|---|---|---|---|---|---|
| **A** | 1,266,816 | d=128, 4층, head 4 | 26M | 1.2분 | **2.0423** |
| **B** | 7,450,880 | d=256, 8층, head 8 | 147M | 23.5분 | **1.3867** |
| **C** | 26,960,384 | d=512, 8층, head 8 | 545M | 176.6분 | **1.1380** |

## 표준 시작 — 데이터 분포 한복판

> 입력: `Once upon a time`

**A** (1,266,816 params, val 2.042)

```
, there was a little girl named Lily. She liked to play outside with her friends. One sunny day, Lily found a big, round box. Inside the box was a beautiful dress with lots of colors. Lily was very happy. She wanted to show her friends to come and play with her friends. Lily's dad was proud of Lily. He said, “Thank you, Lily! Now I would like it." Lily was happy to have a new dress too. They played with the dress all day. The sun was shining, and the ice was very happy.<|endoftext|> Once upon a time, there was a little girl named Mia. She
```

**B** (7,450,880 params, val 1.387)

```
, there was a little girl called Molly. Molly liked to play outside with her friends. One day, when they were playing, Molly noticed something strange. A big bird had moved across the sky! Molly was frightened. She wanted to get away, so she ran to her mom. Molly's mom said to her, "It's okay, sweetheart! That bird must be looking for a place to belong." Molly looked up and saw a friendly bird perched on a branch. She said she wanted to keep it. Molly looked up at the bird and said, "It's nice to own it." The bird chirp
```

**C** (26,960,384 params, val 1.138)

```
, there was a little girl named Sue. Sue liked to play outside with her friends. One day, Sue found a small, red box. Inside the box, she found a pretty ornament. It was round and had lots of colors. Sue showed the ornament to her friends. They all thought it was very nice. Sue wanted to know what would happen if she put the ornament in her pocket. She thought it would be a fun game. So, Sue put the ornament in her pocket and went to play with her friends. While Sue was playing, something unexpected happened. The ornament started to glow! It made a big, red ball out of
```

## 등장인물과 인과

> 입력: `The dragon was very hungry.`

**A** (1,266,816 params, val 2.042)

```
The dragon knew the king was going to eat. The dragon and the dragon became friends and played together every day. The dragon and the dragon were very happy. The dragon had played with its friends, and they all had lots of fun and became the best of friends.<|endoftext|> Once upon a time, there was a little girl named Mia. Mia had a friend named Sam. They liked to play together. One day, Mia's mom said, "I have a gift for you." Mia and Sam went to the house. They saw a big box of toys. They wanted to take them and play with them. Mia said, "Look,
```

**B** (7,450,880 params, val 1.387)

```
The dragon knew he had to go away. He flew high and low, looking for food. He saw the little birds and the big bear. The little birds were scared. The big bear tried to catch them, but he could not. He went to a big tree with lots of leaves. He fell into the woods. The little birds were sad. They did not know how to get away from the big bear. They needed help. The little birds told the big bear to remain awake. They flew to the big bear. They told him about the big bear. The big bear was sad too. He wanted to catch the little birds
```

**C** (26,960,384 params, val 1.138)

```
The dragon saw the little girl and the little girl. The dragon wanted to eat them too. The dragon thought of a plan. The dragon would strike the little girl and take her away. The little girl was scared, but she also wanted to be brave. So, the dragon took her away. The little girl and the dragon met in the cave. They became friends. The dragon was not hungry anymore. The dragon was not a dragon, but a friendly dragon. They lived happily in the big cave.<|endoftext|> Once upon a time, there was a little boy named Tim. Tim loved to play outside with his friends. One day, he
```

## 문맥 이어받기

> 입력: `Tom and Lily found a box. Inside the box was`

**A** (1,266,816 params, val 2.042)

```
a shiny shiny toy. The box had a secret! Tom and Lily were very happy. They opened the box and found a note under a tree. The note said, "Thank you for helping me. Now, I will give you a new toy to play with." Tom and Lily took turns playing with their new toy. They played together in the box all day. They learned that sharing and being kind and sharing is more fun.<|endoftext|> Once upon a time, there was a big bear and a small rabbit. The bear was a very serious bear. He liked to play all day. One day, the bear met a small bear. The bear
```

**B** (7,450,880 params, val 1.387)

```
a big, shiny necklace. They were very happy and played with the necklace all day. At the end of the day, Lily wore the necklace. It made her feel important. She was glad she had found a new friend in the box.<|endoftext|> One day, a little girl named Lily went to the park. It was a warm day, and she was very happy. She saw a big bench near the park. Lily thought it looked like a fun place to sit and play. Lily sat on the bench and started to think. She saw a big dog. The dog was not happy. The dog barked at Lily, her face turned red.
```

**C** (26,960,384 params, val 1.138)

```
a shiny coin. "Wow, look at this!" Tom said. "It's so pretty!" Lily nodded. "It's mine. You took it from the box!" Tom was angry. He wanted the coin. He ran to the box and tried to grab it. But Lily was faster. She caught Tom's arm and pulled the coin from his hand. "Hey, give it back!" Tom said. "That's not fair!" Lily did not listen. She held the coin tight and said, "No, it's mine! I found it!" Tom and Lily started to pull and tug on the coin
```

## 감정과 이유

> 입력: `Sara was sad because`

**A** (1,266,816 params, val 2.042)

```
of the big fish. The fish felt better. They played together in the peaceful pond all day. One day, the big fish and the little fish did not have a new friend. They knew something unexpected happened. The big fish came and said, "Maybe someone should say" to the big fish. The big fish swam away, and the big fish was very sad. From that day on, the big fish always helped the little fish.<|endoftext|> Once upon a time, there was a cute little girl named Sue. Sue loved to sit on the grass and look at the trees. She wanted to draw a pretty picture, her favorite is on a
```

**B** (7,450,880 params, val 1.387)

```
her friend did not believe how hard it was. She thought and thought about how her friend was learning the value of learning and how to do different things. Then, she had an idea. She would ask her friend to teach her how to do the things she believed. Sara went to her friend and showed him how to do the things she learned. They learned how to make up new things and have fun together. They learned that being different was fun and important too. And from that day on, Lucy was not as big or heavy as before. She was the happiest girl in the world.<|endoftext|> Once upon a time, there was a
```

**C** (26,960,384 params, val 1.138)

```
her friend did not believe her. The next day, Sara woke up and saw her mom with the brush. She felt happy and proud of her mom. Her mom smiled and played her song for Sara. Sara knew that her mom was doing something good for her. From that day on, Sara knew that her mom believed in her, just like the tree did.<|endoftext|> Once upon a time, there was a little cat named Fluffy. Fluffy was very independent. She liked to do things by herself. One day, Fluffy saw a big tree with a lot of birds. She wanted to play with the birds. Fluffy jumped and ran to the tree
```

## 학습 데이터 밖의 문장

> 입력: `The quarterly revenue forecast indicates that`

**A** (1,266,816 params, val 2.042)

```
the other puisated, all of the other animals in the world. She was amazed at the bad oxygen, but she loved to be the best friend ever. The end.<|endoftext|> Once upon a time in a small garden, there was a little girl named Lily. She loved to play with her ball in the garden. One day, Lily met a big ugly frog named Doggy. "Hi, Doggy! Why did you catch the frog?" asked Kitty. Doggy said, "I want to catch you, Kitty. Let's play together!" They both looked at each other and laughed. They played with the ugly frog,
```

**B** (7,450,880 params, val 1.387)

```
the boy knew the time. From that day on, the boy and the microwly became the best of friends.<|endoftext|> Once upon a time, there was a big dog named Max. Max had a ball that he loved to play with. One day, he went out to play with his ball. Max played and played, and he was very dirty. He kicked the ball in the puddle. The ball went into the puddle and got dirty. Max was sad because his ball was dirty and dirty. Max's friend, a little bird, saw him and said, "Max, can I help you clean your ball?" Max was happy to
```

**C** (26,960,384 params, val 1.138)

```
the storm had come. The family ran outside. They were safe. The daughter was so proud of the courage she had been to for so long. She had faced the storm with courage.<|endoftext|> Once upon a time, there was a big tree. Under the tree, there was a little boy named Tim. Tim had a bag with a zip. He liked to put things in his bag. One day, Tim found a big, red apple. He put it in his bag with the zip. Then, he went home with the big, red apple. At home, Tim put the big, red apple in his bag with
```
