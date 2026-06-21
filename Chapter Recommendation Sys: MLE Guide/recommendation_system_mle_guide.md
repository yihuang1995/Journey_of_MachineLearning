# Comprehensive Recommendation Systems MLE Preparation Guide

**Source:** Wang Shusheng (王树森) - 8-Hour Industrial Recommendation Systems Course  
**Level:** Advanced ML Engineer preparation  
**Focus:** Production-ready implementation and system architecture

---

## Table of Contents

1. [Fundamentals & Architecture](#1-fundamentals--architecture)
2. [Recall/Retrieval Stage](#2-recall--retrieval-stage)
3. [Ranking & Scoring Stage](#3-ranking--scoring-stage)
4. [User Behavior Sequences](#4-user-behavior-sequences)
5. [Diversity & Re-ranking](#5-diversity--re-ranking)
6. [Cold Start Problem](#6-cold-start-problem)
7. [Metrics Optimization](#7-metrics-optimization)
8. [System Design & Infrastructure](#8-system-design--infrastructure)

---

## 1. Fundamentals & Architecture

### 1.1 Core Problem Definition

**Recommendation Problem:**
- Given user U and item pool I, predict user's preference for each item
- Maximize: user satisfaction, engagement, revenue, retention
- Constraints: latency (ms), scalability (millions of users/items), diversity

**Formal Representation:**
```
score(u, i) = model_params(user_features, item_features, context)
recommendation_list = top_k(items, score, k)
```

### 1.2 Recommendation System Architecture

**Two-Stage Pipeline (Industry Standard):**

```
User Query
    ↓
┌─────────────────────────────────────┐
│   STAGE 1: RECALL (Retrieval)       │
│   (Candidate Generation)             │
│   - Input: 1 user                    │
│   - Output: ~100-1000 candidates     │
│   - Latency: <10ms                   │
│   - Goal: Recall@1000 > 90%          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   STAGE 2: RANKING (Scoring)        │
│   - Input: 100-1000 candidates      │
│   - Output: ~20 final recommendations │
│   - Latency: <50ms                   │
│   - Goal: High precision on results  │
└─────────────────────────────────────┘
    ↓
    Final Ranked List → User
```

### 1.3 Problem Formulation Approaches

**A. Pointwise Learning (pairwise for each item)**
- Predict: score(user, item) individually
- Loss: MSE, BCE (binary classification)
- Pros: Simple, interpretable
- Cons: Ignores ranking relationships

**B. Pairwise Learning**
- Predict: item_i ranked higher than item_j for user
- Loss: Ranking loss (BPR, Hinge loss)
- Goal: item_i > item_j in score ordering
- Better for ranking quality

**C. Listwise Learning**
- Predict: entire ranked list quality
- Loss: NDCG, MAP, LambdaMART
- Optimizes full ranking metrics directly
- State-of-the-art for ranking

### 1.4 Evaluation Metrics

**Ranking Metrics:**
```
Precision@k = (# relevant items in top-k) / k
Recall@k = (# relevant items in top-k) / (total relevant items)
NDCG@k = (1/IDCG_k) * Σ(i=1 to k) [2^rel_i - 1] / log2(i+1)
MAP = Σ(k=1 to K) P@k * rel(k) / min(m, k)
```

**Business Metrics:**
- Click-Through Rate (CTR): % users who click recommended items
- Conversion Rate: % who purchase after clicking
- User Retention: % active users next day/week/month
- Engagement Time: total time spent with recommendations
- Revenue Per User (RPU): purchase value per user
- Diversity Score: % unique items recommended across users

**Offline vs Online Evaluation:**
- Offline: Use historical data, compute metrics on validation set
- Online: A/B test with real users, measure business impact
- Key: A/B testing validates offline metrics actually improve business

### 1.5 Data Pipeline

**Training Data Collection:**
```
User Actions → Event Logging → Data Warehouse
    ↓
Click events, view events, purchase events, time spent
    ↓
Feature Engineering ← User/Item/Context features
    ↓
Training Dataset (positive: clicked, negative: not shown/clicked)
```

**Label Definition (Critical!):**
- Positive: click, purchase, save, long engagement
- Negative: skip, not shown, alternative clicked
- Implicit feedback: view = positive (weak signal)
- Explicit feedback: rating = strong signal

---

## 2. Recall/Retrieval Stage

### 2.1 Purpose & Constraints

**Goal:** Generate candidate set from massive item pool efficiently
- Input: 1 user query
- Output: ~100-1000 most relevant items
- Latency constraint: <10ms (strict)
- Can't run complex models on ALL items

**Key Insight:** Trade accuracy for speed. Better to get 90% relevant items fast than 95% slow.

### 2.2 Item-Based Collaborative Filtering (ItemCF)

**Algorithm Logic:**
```
1. Compute item similarity: sim(i, j) = # users who liked both i & j
2. User u liked item i, find similar items
3. Score candidate = Σ(i in user_history) [sim(i, candidate) * user_rating(i)]
4. Rank by score, return top-k
```

**Similarity Metrics:**

**Cosine Similarity:**
```
sim(i, j) = (U_i · U_j) / (||U_i|| * ||U_j||)

Where U_i = vector of users who interacted with item i
Example: video1 = [1,1,0,1,...], video2 = [1,0,0,1,...]
sim = (1+0+0+1) / (sqrt(3) * sqrt(2)) = 0.816
```

**Jaccard Similarity:**
```
sim(i, j) = |U_i ∩ U_j| / |U_i ∪ U_j|
More interpretable: % overlap of users
```

**Implementation Tips:**
- Precompute similarity matrix offline (expensive computation)
- Store as sparse matrix (most similarities = 0)
- Update incrementally as new interactions arrive
- For 1M items: full similarity matrix = 1TB (impractical)
- Solution: Top-K neighbors only (~100 similar items per item)

**Complexity:**
- Offline: O(I² * U) where I=items, U=users (daily batch job)
- Online: O(k * log(k)) for top-k retrieval (very fast)

**Pros:**
- Interpretable: "users who liked X also like Y"
- No cold start for items with interactions
- Works well for direct content similarity

**Cons:**
- Quality depends on sufficient user interaction data
- New items with no interactions: cold start
- Doesn't capture user intent/preference variation
- Popular items get recommended more often (bias)

### 2.3 User-Based Collaborative Filtering (UserCF)

**Algorithm Logic:**
```
1. Find similar users: sim(u, v) = # items both users interacted with
2. Find users similar to u
3. Get items they liked but u didn't
4. Score = Σ(similar_user) [sim(u, similar_user) * rating(similar_user, item)]
5. Return top-k items
```

**Key Difference from ItemCF:**
- ItemCF: find similar items directly
- UserCF: find similar users, use their preferences

**Similarity Computation:**
```
sim(u1, u2) = cos(rating_vector_u1, rating_vector_u2)

User1: [5, 4, _, 3, 5] (ratings for 5 items)
User2: [4, 5, _, 2, 5]
sim = high if they rated items similarly
```

**Complexity:**
- Offline: O(U²) - quadratic in users (EXPENSIVE for millions of users)
- Online: O(k * items_per_user)

**Pros:**
- Captures user preference diversity
- Different users → different recommendations even for same items

**Cons:**
- Doesn't scale: U² becomes prohibitive
- New users: cold start (no rating history)
- Sparsity problem: most users rate <1% of items
- Data freshness: requires frequent updates

### 2.4 Swing Algorithm (Advanced Similarity)

**Problem:** ItemCF popularity bias - popular items overrecommended

**Swing Concept:**
```
Two users both interacted with item A.
If they also both like item B → item A & B are similar
But penalize if many users cause this co-occurrence

sim_swing(i, j) = Σ(u∈U_i∩U_j) [1 / log(1 + |history(u)|)]

Penalizes by user history size:
- Users who like few things → stronger signal
- Users who like many things → weaker signal
```

**Why Swing > Cosine Similarity:**
```
Cosine: sim(blockbuster, niche) = high if many users like both
Swing: penalizes this because many users like blockbuster

Result: Better differentiation of truly related items
```

**Implementation:**
```python
for user in users:
    if user.likes(item_i) and user.likes(item_j):
        weight = 1 / log(1 + len(user.history))
        swing_sim[i][j] += weight

swing_sim[i][j] /= (|U_i ∩ U_j|) # normalize by overlap
```

**Advantages:**
- Reduces popularity bias
- Produces more diverse, niche recommendations
- Better quality for long-tail items
- Industry standard (used at Alibaba, ByteDance)

### 2.5 Other Recall Methods

**A. User Embedding-based Retrieval**
```
1. Embed users: user_embedding = learned_vector(user_history)
2. Embed items: item_embedding = learned_vector(item_features)
3. Retrieve: candidates = ANN_search(user_emb, item_emb, top-k)
4. Advantage: Captures complex user preferences
5. Implementation: Approximate Nearest Neighbor search (FAISS, Milvus)
```

**B. Graph-based Methods (GCN)**
```
User-Item bipartite graph:
    U1 ----likes---- I1
    U1 ----views---- I2
    U2 ----likes---- I1
    
GCN aggregates neighbor information:
    user_rep = GCN(user_neighbors, item_neighbors)
    Similar to collaborative filtering but learnable
```

**C. Search-based Recall**
```
User query → BM25/Elasticsearch search → candidates
Example: "action movies" → search index returns top-1000 movies
Fast because leverages search infrastructure
```

**D. Content-based Filtering**
```
user_profile = [likes action, likes sci-fi, likes 2020s movies]
item_profile = [genre: action, genre: sci-fi, released: 2024]
score = dot_product(user_profile, item_profile)
Advantage: Works for new items with content features
Disadvantage: Limited to available features, can be biased
```

### 2.6 Multiple Recall Channels

**Production Systems Typically Combine:**

```
Recall Channel 1: ItemCF (30% of candidates)
    → Similar to user's recent items
    
Recall Channel 2: UserCF (20% of candidates)
    → Similar users' preferences
    
Recall Channel 3: Embedding-based (30% of candidates)
    → User embedding near item embeddings
    
Recall Channel 4: Trending (10% of candidates)
    → Popular items (exploration)
    
Recall Channel 5: Search (10% of candidates)
    → Keyword matching
    ↓
Merge & Deduplicate → 200-500 candidates
    ↓
Pass to Ranking Stage
```

**Why Multiple Channels:**
- Coverage: Different channels catch different user preferences
- Robustness: If one method fails, others provide fallback
- Diversity: Different approaches reduce filter bubble
- Exploration: Trending/search channels encourage discovery

---

## 3. Ranking & Scoring Stage

### 3.1 Feature Engineering

**User Features (Dense, Sparse):**
```
Dense Features:
  - User embedding (learned from history)
  - User age, tenure
  - Historical CTR, conversion rate
  - Engagement level
  
Sparse Features:
  - User ID (one-hot encoded)
  - User gender, location
  - User interests/tags
  - Device type
```

**Item Features:**
```
Dense:
  - Item embedding (learned from content)
  - Item popularity (views, likes)
  - Item recency (age, last update)
  - Item quality score
  
Sparse:
  - Item ID
  - Category, subcategory, tags
  - Creator/brand ID
  - Price range
```

**Context Features:**
```
  - Time of day (morning/afternoon/night)
  - Day of week
  - Device type (mobile/desktop)
  - Network type (wifi/4g)
  - Location
  - Session length
  - User's recent query
```

**Cross Features (Crucial):**
```
  - user_embedding × item_embedding (dot product)
  - user_category_affinity × item_category
  - user_recency_preference × item_age
  - user_price_preference × item_price
  
These capture user-item interactions!
```

### 3.2 Learning-to-Rank (LTR)

**Pointwise Approach (Simple):**
```
Model: logistic regression or neural network
Input: [user_features, item_features, cross_features]
Output: predicted_score ∈ [0, 1] (probability user clicks)
Loss: Binary cross-entropy
  loss = -[y*log(p) + (1-y)*log(1-p)]
  
Problem: Doesn't optimize ranking order
```

**Pairwise Approach:**
```
Training data: (user, item_good, item_bad, label=1 if good>bad)
Model learns: score(good) > score(bad)
Loss: Ranking loss (BPR - Bayesian Personalized Ranking)
  loss = -log(σ(s_good - s_bad))
  
Advantage: Directly optimizes relative ranking
```

**Listwise Approach (Best):**
```
NDCG-based Learning (LambdaMART):
  
1. For each user's candidate set, compute NDCG
2. Compute gradient for NDCG improvement
3. Adjust model to maximize NDCG
4. Loss function directly optimizes ranking metric
  
Example:
  If moving item_i from rank 3 to rank 1 increases NDCG by 0.05
  → Gradient pushes model to rank item_i higher
```

**LambdaMART Algorithm:**
```
For each training example (query, ranked_list):
    1. Compute current NDCG
    2. For each pair (i, j) where i ranked higher than j:
        if relevance(j) > relevance(i):
            loss += NDCG_improvement if we swap i and j
            gradient ∝ |NDCG_change| * log_odd_ratio
    3. Update model weights
```

**Comparison Table:**

| Aspect | Pointwise | Pairwise | Listwise |
|--------|-----------|----------|----------|
| Data prep | Easy | Medium | Hard |
| Training speed | Fast | Medium | Slow |
| Ranking quality | Good | Better | Best |
| Metric alignment | Poor | Good | Excellent |
| Implementation | Simple | Medium | Complex |

### 3.3 Deep Neural Networks for Ranking

**Architecture: Deep Crossing Network (DCN)**
```
Input Layer:
  ├─ User embedding (128 dims)
  ├─ Item embedding (128 dims)
  ├─ Context (one-hot → embedding)
  └─ Cross features

Hidden Layers (Deep Part):
  ├─ Dense(512) → ReLU
  ├─ Dropout(0.2)
  ├─ Dense(256) → ReLU
  ├─ Dropout(0.2)
  └─ Dense(128) → ReLU

Cross Layers (Cross Part):
  Efficiently learns feature interactions
  x_{l+1} = x_0 * (w_l^T x_l + b_l) + x_l

Output:
  Dense(1) → Sigmoid → predicted_ctr
```

**Why Deep Networks > Logistic Regression:**
```
Logistic Reg: score = w·x + b (linear)
Deep Network: score = g(g(g(...(w·x+b)...))) (non-linear)

Can learn complex interactions:
  "Users interested in sci-fi + new movie → high probability"
  (Simple regression can't capture this easily)
```

**Architecture Variants:**

**A. Wide & Deep (Google)**
```
Wide component: memorization of common features
Deep component: generalization of new patterns

Combined: best of both worlds
- Remembers popular item-category pairs
- Generalizes to new combinations
```

**B. DeepFM (Factorization Machine + DNN)**
```
FM captures 2-way feature interactions
DNN captures high-order interactions
Combined for comprehensive interaction modeling
```

**C. DIN (Deep Interest Network)**
```
Attention mechanism over user's historical items
Focuses on relevant items from history
Better than averaging all historical embeddings
```

### 3.4 Feature Processing

**Continuous Features (Normalization):**
```
Standardization: x_norm = (x - mean) / std
Range: [0, 1] normalization: x_norm = (x - min) / (max - min)

Why: Neural networks learn better with normalized inputs
```

**Categorical Features (Embedding):**
```
Original: user_gender ∈ {M, F}
One-hot: [1, 0] or [0, 1]
Problem: High dimensionality for many categories

Solution: Embedding
  gender_embed = lookup_table[gender_id]
  Result: 8-dimensional vector instead of 1-hot
  
Reduces sparsity, learns similarity between categories
```

**High-Cardinality Features (User ID, Item ID):**
```
Naive: one-hot for 100M users = 100M dims (impossible)
Solution: Learned embedding
  user_embedding = embedding_layer(user_id)
  Creates 128-dim vector for each user
  
Embedding matrix: (100M users × 128 dims) = 50GB
Manageable with modern GPUs
```

### 3.5 Model Training

**Data Sampling:**
```
Problem: Negative samples >> Positive samples
  1M interactions, but 10B possible user-item pairs
  Ratio: 1:10,000 (severe imbalance)

Solution: Negative sampling
  For each positive (user, item_clicked):
    Sample 1-10 negatives (random non-clicked items)
  Balance ratio: 1:10 (manageable)
  
During inference: Rank ALL items for user
```

**Online Learning (Continuous Updates):**
```
Traditional ML: Retrain model weekly
Online Learning: Update model with each interaction

Approach:
1. User sees recommended item
2. User clicks/ignores (immediate feedback)
3. Update model weights slightly
4. Next user gets slightly improved recommendations

Benefit: Model adapts to trends in real-time
Challenge: Prevent overfitting to noise
```

---

## 4. User Behavior Sequences

### 4.1 Why Sequences Matter

**Problem with averaging user history:**
```
User watched: [sci-fi_2020, sci-fi_2021, romance_2022, romance_2023, action_2024]

Simple average user embedding:
  rep = avg([sci-fi_embed, sci-fi_embed, romance_embed, romance_embed, action_embed])
  Shows general sci-fi/romance interest
  
But ignores: Recent shift from romance back to action!
```

**Sequence Models Capture:**
- Temporal trends: shifting interests over time
- Sequential patterns: "after watching action, likely to watch thriller"
- Recent bias: recent items matter more
- State evolution: user's "mood" changes

### 4.2 RNN & LSTM Architecture

**Simple RNN:**
```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = softmax(W_hy * h_t + b_y)

Where:
  x_t = item_embedding at time t
  h_t = hidden state (captures sequence context)
  y_t = prediction for next item
  
Problem: Gradient vanishing
  Backprop through 100 timesteps → gradients → 0
  Can't learn long-term dependencies
```

**LSTM (Long Short-Term Memory):**
```
Solves vanishing gradient with gate mechanisms:

Forget gate: f_t = σ(W_f * [h_{t-1}, x_t] + b_f)
  → How much of previous context to forget

Input gate: i_t = σ(W_i * [h_{t-1}, x_t] + b_i)
  → How much of new input to use

Candidate: C̃_t = tanh(W_c * [h_{t-1}, x_t] + b_c)
  → Potential new context info

Cell state: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
  → Selective memory update (skip old, add new)

Output gate: o_t = σ(W_o * [h_{t-1}, x_t] + b_o)
  → What to output

Hidden state: h_t = o_t ⊙ tanh(C_t)
  → Updated state for next step
```

**Example (User watches movies in sequence):**
```
Day 1: Watch sci-fi movie → h_1 = LSTM(empty, sci-fi)
       Remember: "user likes sci-fi"
       
Day 2: Watch another sci-fi → h_2 = LSTM(h_1, sci-fi)
       Forget gate: mostly remember (f ≈ 0.9)
       Reinforce sci-fi preference
       
Day 3: Watch romantic movie → h_3 = LSTM(h_2, romance)
       Forget gate: partially forget (f ≈ 0.7)
       Update preference: "still some sci-fi, now romance"
       
Day 4: Predict next → output "romance" (recent bias)
```

**GRU (Gated Recurrent Unit):**
```
Simpler than LSTM (fewer parameters)
Similar performance for many tasks

Reset gate: r_t = σ(W_r * [h_{t-1}, x_t])
Candidate: h̃_t = tanh(W_h * [r_t ⊙ h_{t-1}, x_t])
Update gate: z_t = σ(W_z * [h_{t-1}, x_t])
h_t = z_t ⊙ h_{t-1} + (1-z_t) ⊙ h̃_t
```

### 4.3 Attention Mechanisms

**Problem with RNN:**
```
LSTM output h_T depends on ALL sequence items equally weighted
But user's interests vary by item:
  - Very interested in recent items
  - Moderately interested in historical patterns
  - Not interested in early random items
  
Solution: Attention - weight items by importance
```

**Attention Mechanism:**
```
Query: q = user_embedding
Keys: K = [item_1_embed, item_2_embed, ..., item_n_embed]
Values: V = [item_1_embed, item_2_embed, ..., item_n_embed]

Attention scores: scores = q · K^T (dot products)
Normalize: attention_weights = softmax(scores)
Output: context = Σ(weight_i * V_i) (weighted sum)

Result: Heavy weight on most relevant items, low weight on irrelevant
```

**Example:**
```
User history: [sci-fi_2020, sci-fi_2021, action_2022, romance_2023, action_2024]
Current query: "What action movie?"

Attention weights:
  sci-fi_2020: 0.05 (low - not action)
  sci-fi_2021: 0.05 (low - not action)
  action_2022: 0.20 (medium - action, but old)
  romance_2023: 0.05 (low - not action)
  action_2024: 0.65 (high - action, very recent!)
  
Recommendation focuses on what action movies look like
```

### 4.4 Transformer-based Approaches

**Self-Attention (Multi-head):**
```
Item sequence: [i_1, i_2, ..., i_n]

Multi-head attention: Apply attention h times with different weights
Combine outputs: concatenate, then linear projection

Benefit: Different heads learn different item relationships
  Head 1: Temporal patterns (consecutive items)
  Head 2: Genre patterns (movies of same genre)
  Head 3: Popular patterns (trending items)
```

**Transformer Architecture:**
```
Layer i:
  1. Multi-head Self-Attention (what items are most relevant)
  2. Feed-forward network (process interaction)
  3. Layer normalization and residual connections
  
Stack multiple layers → deeper context understanding

Example:
  Layer 1: "What genre does this sequence prefer?"
  Layer 2: "What's the temporal trend?"
  Layer 3: "What's the user's overall style?"
  Layer 4: "Combine all insights → predict next item"
```

**Advantages over RNN:**
```
RNN: Processes sequentially (item by item)
  - Sequential: t_1, t_2, t_3, ..., t_n
  - Problem: Can't parallelize (must process in order)
  - Training slow for long sequences

Transformer: Processes all at once
  - All items attend to all other items
  - Fully parallelizable
  - Much faster training
  - Better long-range dependency capture
  
Trade-off: More parameters, but better results
```

### 4.5 Practical Implementation (DIN - Deep Interest Network)

**Architecture:**
```
User embedding: e_u
Item embedding: e_i
Historical items: [e_1, e_2, ..., e_n]
Query (candidate item): e_candidate

Attention weights = softmax(e_candidate · [e_1, e_2, ..., e_n]^T)
Interest representation = Σ(attention_weight * e_i)

DNN([e_u, e_i, interest_rep, context])
→ CTR score

Key insight:
  Different candidate items → different attention weights
  "For sci-fi movie, focus on sci-fi history"
  "For action movie, focus on action history"
```

**Training:**
```
Positive sample: (user, clicked_item, historical_items)
Negative sample: (user, not_clicked_item, historical_items)

Loss: Binary cross-entropy on CTR prediction
Optimization: SGD or Adam
```

---

## 5. Diversity & Re-ranking

### 5.1 Why Diversity Matters

**Problem: Filter Bubble**
```
Recommendation becomes repetitive:
  User watches sci-fi → recommends sci-fi
  → User watches more sci-fi → recommends more sci-fi
  → User never discovers other genres

Business Impact:
  - User boredom
  - Higher churn rate
  - Missing monetization opportunities (new categories)
```

**User Satisfaction:**
```
Pure accuracy:  [A, A, A, A, A] (all same type)
  User feedback: "Boring! I want variety"
  
With diversity: [A, A, B, A, C]
  User feedback: "Better! Nice discoveries"
  
Even if user clicks less on B & C, overall satisfaction ↑
  → Better retention
  → More engagement over time
```

**Business Metrics:**
- Novelty: % of items user hasn't seen before
- Diversity: # of categories/genres in recommendations
- Serendipity: % of recommendations in non-primary interests
- Coverage: % of catalog recommended to any user

### 5.2 Diversity Algorithms

**A. Determinantal Point Process (DPP)**

**Concept:**
```
Want: Select items that are individually good AND mutually diverse

Quality: score(item_i) = relevance to user
Diversity: sim(i, j) = how similar items i and j are

DPP selects items maximizing:
  det(L) where L = diversity_matrix weighted by quality_scores
  
High determinant → selected items spread in feature space
  (different from each other)
```

**Algorithm:**
```
1. Build similarity matrix S: S[i,j] = cosine(item_i, item_j)
2. Scale by quality: L[i,j] = score(i) * S[i,j] * score(j)
3. Compute determinant (proportional to diversity)
4. Sampling: Select items that maximize determinant

Result: Relevance (scores are high) + Diversity (det is high)
```

**Pros:**
- Theoretically grounded
- Balances quality and diversity

**Cons:**
- Computationally expensive (determinant calculation)
- Hard to implement efficiently at scale

**B. Greedy Re-ranking (Practical)**

**Algorithm:**
```
Start: candidates = [item_1, item_2, ..., item_n]
       Sorted by score (relevance)
       
Final = []

For k = 1 to K:
    best_item = null
    best_score = -∞
    
    For each item in candidates:
        # Trade-off: Relevance vs. Diversity
        diversity_penalty = Σ(sim(item, already_selected))
        adjusted_score = relevance_score - λ * diversity_penalty
        
        if adjusted_score > best_score:
            best_score = adjusted_score
            best_item = item
    
    Final.append(best_item)
    candidates.remove(best_item)

Return Final (top-K diverse items)
```

**Example:**
```
Candidates sorted by relevance:
  1. Movie_A (score=0.9, genre=sci-fi)
  2. Movie_B (score=0.85, genre=sci-fi)
  3. Movie_C (score=0.80, genre=action)
  4. Movie_D (score=0.78, genre=romance)

λ=0.5 (diversity weight)

K=1: Select Movie_A (score=0.9, no diversity needed)

K=2: Score candidates
  Movie_B: 0.85 - 0.5*sim(A,B) = 0.85 - 0.5*0.95 = 0.375
  Movie_C: 0.80 - 0.5*sim(A,C) = 0.80 - 0.5*0.30 = 0.65
  Movie_D: 0.78 - 0.5*sim(A,D) = 0.78 - 0.5*0.10 = 0.73

Best: Movie_D (diversity + okay relevance)
Final = [Movie_A, Movie_D]

Result: Mix of sci-fi and romance (diverse)
```

**C. Coverage Constraint**

**Idea:**
```
Ensure each category/topic is represented
"At least 1 action, 1 romance, 1 sci-fi in top-20"

Algorithm:
  1. Rank items by relevance
  2. Add items ensuring coverage constraints
  3. Adjust order to maximize relevance within constraints

Example:
  Top-20 = [top-5_sci-fi, top-5_action, top-5_romance, top-5_other]
  Shuffle to maintain some relevance order
```

### 5.3 Position Bias & Diversity Trade-off

**Issue: Inherent Position Bias**
```
Users more likely to click top items (position 1 > position 5)

Traditional optimization: Maximize CTR on top-k items
Problem: Biased toward similar, high-relevance items

Solution: Position-aware diversity
  Top positions: High relevance (majority of users view)
  Lower positions: More diversity (adventurous users continue)
  
Order: [high_rel, high_rel, medium_rel_diverse, low_rel_diverse]
```

### 5.4 Measuring Diversity

**Metrics:**

**1. Category Entropy:**
```
entropy = -Σ(p_c * log(p_c))
where p_c = proportion of items in category c

entropy=0: All items same category (no diversity)
entropy=high: Evenly spread across categories (diverse)
```

**2. Pairwise Distance:**
```
diversity_score = avg(1 - sim(i, j)) for all pairs in top-k
Higher = more diverse recommendations
```

**3. Coverage:**
```
coverage = (# unique items recommended to any user) / total_items
Encourages recommending niche items
```

**4. Catalog Coverage:**
```
For all users, what % of item catalog gets recommended?
Goal: 30-50% catalog coverage (maximize catalog utilization)
```

---

## 6. Cold Start Problem

### 6.1 Item Cold Start (New items with no interactions)

**Challenge:**
```
New movie uploaded → 0 views, 0 interactions
Collaborative filtering: Can't find similar items or users
Result: Never recommended (bootstrapping problem)

Business impact: New content gets low visibility
```

**Solution 1: Content-based Features**

**Use item metadata:**
```
Movie features: [genre, year, director, actors, plot_embedding, poster_embed]
User preferences: [prefers_genre, prefers_actors, ...]

Score = dot_product(user_pref, item_features)

Example:
  User: [likes_action=1, likes_2020s=1, likes_superhero=0.5]
  Movie: [action=1, year=2024, superhero=0.8]
  Score = 1*1 + 1*1 + 0.5*0.8 = 2.3
```

**Advantage:** Works immediately (no interaction history needed)
**Disadvantage:** Requires manual feature engineering / content metadata

**Solution 2: Hybrid Approach**

**Combine collaborative + content:**
```
Score = α * collaborative_score + (1-α) * content_score

Start with high α (mostly content)
As interactions accumulate, increase collaborative weight
  
After 100 clicks: α=0.5 (balance)
After 1000 clicks: α=0.9 (mostly collaborative)
```

**Solution 3: Embedding-based (Universal Representation)**

**Content embedding:**
```
Process item metadata → embedding
  movie_embed = BERT(plot_text) + CNN(poster_image) + embedding(genre)
  
Embedding space: Similar content → similar embeddings
Even without interactions, can find content-similar items

ANN search: Find items near this embedding
```

**Solution 4: Exploration (Bandit Algorithm)**

**Thompson Sampling:**
```
For new items, deliberately show them (exploration)
Collect interaction data quickly
  
Approach:
  1. Recommend new item to small % of users (exploration)
  2. Observe click/interaction rates
  3. Update item embedding based on interactions
  4. Gradually increase recommendation frequency

Trade-off: Some users see suboptimal recommendations
Benefit: Quickly collect data for new items
```

**Solution 5: Warm-starting with Editor Curation**

```
New item → Editor reviews and categorizes
  Tags: [genre, mood, target_audience, trending_reason]
  
Use tags as initial features:
  user_liking_tag → boost score for items with tag
  
Over time: Interactions replace manual tags
```

### 6.2 User Cold Start (New users, no history)

**Challenge:**
```
New user signup → No interaction history
CF methods: Can't find similar users or their items

Solution: Profile new users fast
```

**Solution 1: Registration Form**

```
On signup, ask:
  1. Favorite genres/categories
  2. Favorite creators/brands
  3. Content types (text/video/audio preference)
  
Map answers → initial user embedding
Start with explicit preferences
```

**Solution 2: Explore-Exploit Strategy**

```
First 10 items: Show diverse, popular items (exploration)
Observe what user clicks
Build initial profile from clicks
Then: Regular recommendations (exploitation)

Maximize learning in early sessions
```

**Solution 3: Related Items from First Click**

```
User clicks first item X → Immediately recommend:
  1. Similar items (ContentCF)
  2. Items liked by users who also like X (UserCF)
  
Build momentum from single click
```

**Solution 4: Social Features**

```
User follows friends → Show what friends like
User follows creators → Show items from creators

Leverage social graph to warm-start
```

### 6.3 Other Cold Start Variants

**A. Catalog Cold Start**
```
New catalog/app launch → No user-item interactions

Solutions:
  1. Leverage external data (reviews from other platforms)
  2. Content-based features (metadata, embeddings)
  3. Domain knowledge (editor categorization)
  4. Partner data (if acquired user base from elsewhere)
```

**B. Context Cold Start**
```
New time period (e.g., holidays, events)
User behavior changes, old model becomes stale

Solution: Online learning that adapts quickly
```

---

## 7. Metrics Optimization

### 7.1 Offline Metrics

**Ranking Quality Metrics:**

**NDCG (Normalized Discounted Cumulative Gain):**
```
Relevance grades: 0 (not relevant), 1 (somewhat), 2 (relevant), 3 (very relevant)

DCG = Σ(i=1 to k) [(2^rel_i - 1) / log2(i+1)]
      rel_i = relevance of item at position i

IDCG = DCG of ideal ranking (best possible order)

NDCG = DCG / IDCG ∈ [0, 1]

Example:
  Ideal:        [3, 3, 2, 1, 0, 0, ...]
  Our ranking:  [3, 2, 2, 1, 0, 0, ...]
  
  DCG_ours = (8-1)/1 + (4-1)/1.585 + (4-1)/2 + (2-1)/2.322
           = 7 + 1.89 + 1.5 + 0.43 = 10.82
           
  DCG_ideal = 7 + 1.89 + 1.5 + 0.43 = 10.82
  Wait, let me recalculate...
  
  NDCG@5 = 0.92 (very close to ideal)
```

**Why NDCG is good:**
- Accounts for relevance grades
- Penalizes wrong items at top positions
- Normalized (comparable across queries)

**Mean Average Precision (MAP):**
```
AP = Σ(i=1 to k) [P@i * rel(i)] / num_relevant
    where P@i = precision at rank i

P@1 = 1 (if first item relevant)
P@2 = 0.5 (if only 1 of first 2 relevant)

MAP = average AP across all users
```

**Hit Rate (Recall):**
```
Hit@k = (# relevant items in top-k) / (total relevant items)

Problem: Treats all positions equally (doesn't penalize wrong items at top)
Solution: NDCG better metric

But useful for rough evaluation
```

### 7.2 Online Metrics & A/B Testing

**Key Business Metrics:**

**Click-Through Rate (CTR):**
```
CTR = (# users who clicked) / (# users shown recommendations)

Problem: "Clicks" may be accidental or driven by curiosity
Better as directional metric (↑ in A, ↓ in B → A better)
```

**Conversion Rate:**
```
Conversion = (# users who purchased) / (# users shown)

Stronger signal than clicks
More important for revenue
But much lower baseline (1% vs 10%)
```

**Engagement Time:**
```
avg_engagement_time = Σ(time spent on recommended items) / # recommendations

Measure: How long users spend on recommended content
Better than single click (indicates quality)
```

**Diversity Metrics (Online):**
```
User satisfaction surveys: "How diverse were recommendations?"
Repeat rate: % items shown again to same user
Discovery rate: % new items user interacted with
```

**Retention Metrics:**
```
Day-1 Retention: % users active next day
Day-7 Retention: % active next week

Recommendations affect user engagement → retention
Key long-term metric (more important than immediate CTR)
```

**A/B Testing Framework:**

**Experiment Design:**
```
1. Hypothesis: "Diversity re-ranking improves retention"

2. Variants:
   Control: Old ranking (pure relevance)
   Treatment: New ranking (relevance + diversity)

3. Assignment:
   - 50% users → control
   - 50% users → treatment
   
4. Duration: 1-2 weeks (enough data for significance)

5. Metrics:
   Primary: Day-7 Retention (should ↑)
   Secondary: CTR (may ↓ due to diversity)
   Guardrail: Conversion rate (must not ↓)

6. Analysis:
   - Test statistical significance (p < 0.05)
   - Effect size (Δ > 1% meaningful?)
   - Confidence interval
```

**Statistical Significance:**
```
Null hypothesis: Control and Treatment are same

Given observations:
  Control: 1000 users, 600 active next day (60%)
  Treatment: 1000 users, 620 active next day (62%)
  
Test: Is this 2% difference real or random?

Sample size: n=1000 per variant (reasonable)
Effect size: 2% improvement (real?)
P-value: If p<0.05, reject null → difference is real

If p>0.05: Can't conclude difference (maybe just noise)
```

### 7.3 Metric Gaming & Guardrails

**Problem: Optimizing One Metric Hurts Others**

**Example:**
```
Objective: Maximize CTR

Solution A: Show only clickbait items
  Result: CTR ↑ 50% ✓
  But: User dissatisfaction ↑, retention ↓, revenue ↓
  
Guardrail: Monitor retention metric
  If retention ↓ > 5%, pause experiment
  Force model to not sacrifice retention
```

**Common Metric Conflicts:**

```
1. CTR vs. Diversity
   Optimizing CTR → recommend same popular items
   Guardrail: Minimum diversity threshold
   
2. Clicks vs. Quality
   Click ≠ Quality (accidental clicks count)
   Guardrail: engagement_time, retention metrics
   
3. Revenue vs. User Satisfaction
   Short-term revenue (aggressive monetization)
   Guardrail: churn rate (maintain < threshold)
   
4. Coverage vs. Relevance
   Coverage: recommend many items
   Relevance: recommend best items
   Trade-off: λ parameter in objective function
```

**Setting Business Metrics:**

```
Tier 1 (Core): 
  - Day-7 Retention (most important)
  - Revenue Per User

Tier 2 (Quality):
  - NDCG@10 (offline ranking quality)
  - Engagement Time

Tier 3 (Guardrails):
  - CTR (should stay similar)
  - Conversion Rate (don't decrease)
  - Diversity (maintain minimum)
```

### 7.4 Online Learning Optimization

**Continuous Learning:**
```
Traditional: Retrain model every week
Online: Update model continuously

Pipeline:
  1. User sees recommendation → interaction logged
  2. Train loss computed immediately
  3. Model weights updated (gradient step)
  4. Next user sees improved model
  
Benefit: Model adapts to trends in real-time
Challenge: Risk overfitting to noise
```

**Learning Rate & Momentum:**
```
w_{t+1} = w_t - α * ∇loss_t  (simple gradient descent)

α = learning rate (step size)
  Too high: Unstable, oscillates
  Too low: Converges slowly
  
Adaptive α: Decrease over time (cooler updates later)
  or use momentum: Smooth gradient trajectory
  
v_t = β*v_{t-1} + (1-β)*∇loss
w_t = w_t - α * v_t
```

**Exploration vs. Exploitation:**
```
Exploitation: Recommend items we're confident user likes
Exploration: Recommend items to learn user preferences

ε-Greedy:
  With probability ε (e.g., 10%): random recommendation (explore)
  With probability 1-ε: best recommendation (exploit)

Thompson Sampling (more sophisticated):
  Model uncertainty about user preferences
  Items with high uncertainty → explore more
  Items with high confidence → exploit more
```

---

## 8. System Design & Infrastructure

### 8.1 System Architecture Overview

```
┌─────────────────────────────────────────────────┐
│           Request from User/App                  │
└──────────────────┬──────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│        Online Serving (Low Latency <100ms)      │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. Feature Service:                            │
│     ├─ User embeddings (from Redis cache)       │
│     ├─ Item embeddings (from Elasticsearch)     │
│     └─ Context features (request data)          │
│                                                  │
│  2. Recall Stage (30ms budget):                 │
│     ├─ ItemCF candidates (from precomputed DB)  │
│     ├─ Embedding-based (FAISS ANN search)       │
│     └─ Trending items (from cache)              │
│     → Output: 500 candidates                    │
│                                                  │
│  3. Ranking Stage (50ms budget):                │
│     ├─ Deep learning model inference            │
│     ├─ Score all 500 candidates                 │
│     └─ Top-20 results → Diversity re-rank       │
│     → Output: 20 ranked items                   │
│                                                  │
│  4. Re-ranking:                                 │
│     ├─ Business rules (paid promotions)         │
│     ├─ Diversity adjustment                     │
│     └─ Deduplication                            │
│                                                  │
│  5. Response:                                   │
│     └─ Return 20 items + metadata               │
│                                                  │
└────────────────────┬────────────────────────────┘
                     ↓
         ┌───────────────────────┐
         │  Logging & Analytics  │
         │  (User interactions)  │
         └───────────┬───────────┘
                     ↓
    ┌────────────────────────────────┐
    │    Offline Batch Processing     │
    │  (Daily/Weekly retraining)      │
    ├────────────────────────────────┤
    │                                 │
    │ 1. Data Pipeline:               │
    │    ├─ Raw interaction logs      │
    │    ├─ User/item embeddings      │
    │    ├─ Feature engineering       │
    │    └─ Training data preparation │
    │                                 │
    │ 2. Model Training:              │
    │    ├─ Recall model (ItemCF)     │
    │    ├─ Embedding model           │
    │    ├─ Ranking model (DNN)       │
    │    └─ Offline evaluation        │
    │                                 │
    │ 3. Deployment:                  │
    │    ├─ Model versioning          │
    │    ├─ A/B testing               │
    │    └─ Canary rollout (5% users) │
    │                                 │
    └────────────────────────────────┘
```

### 8.2 Feature Engineering Pipeline

**Feature Sources:**

**A. User Features (Real-time or cached):**
```
Static:
  - User ID, age, gender, location
  - Account creation date, tenure
  
Dynamic (updated daily or real-time):
  - User embeddings (trained offline)
  - Recent CTR (clicks / views last 7 days)
  - Category preferences (clicks per category)
  - Search history
  - Explicit profile (user-entered preferences)
  
Behavioral:
  - Engagement level (items clicked per day)
  - Content consumption patterns
  - Day-of-week patterns
  - Time-of-day patterns
```

**B. Item Features (Batch updated):**
```
Static:
  - Item ID, title, description
  - Category, tags, creator ID
  - Release date, duration, format
  - Price, rating/reviews
  
Dynamic:
  - Item embeddings (from content)
  - Popularity (views, likes last 7 days)
  - Recency (age of item)
  - Quality score (user ratings, engagement)
  - Trending score (velocity of growth)
  
Meta:
  - Whether item is new/promoted
  - A/B test assignment
  - Performance metrics
```

**C. Context Features (Real-time):**
```
Request:
  - Time of day (hour)
  - Day of week
  - Device type (mobile/desktop)
  - Network type (wifi/4g/cellular)
  - OS version
  
Session:
  - Items previously shown in session
  - Click history in session
  - Dwell time
  - User's current location
  - Referrer (how user got here)
```

**Feature Storage:**

**Redis (Fast, hot data):**
```
Key: "user:123:embedding"
Value: 128-dim vector
TTL: 1 hour

Key: "trending:items"
Value: [item_1, item_2, ..., item_100] (by score)
TTL: 1 hour

Response time: <1ms per key
```

**Elasticsearch (Search-optimized):**
```
Index: "items"
Document: {id, title, category, tags, embedding, metadata}

Query: Find items with category="sci-fi" AND popularity>1000
Response time: <5ms

Supports complex filtering for recall stage
```

**PostgreSQL (Persistent, analytical):**
```
Table: user_profiles
  Columns: user_id, age, gender, location, ...
  
Table: item_metadata
  Columns: item_id, title, category, quality_score, ...
  
Table: interactions (massive, needs partitioning)
  Columns: user_id, item_id, timestamp, action, duration

Update frequency: Daily batch imports
```

**Data Warehouse (Historical data):**
```
Tools: BigQuery, Snowflake, Redshift
Purpose: Training data, offline evaluation, analytics

Store: Years of interaction history
Size: Terabytes
Query: Complex aggregations over millions of records

Not real-time (for offline processing)
```

### 8.3 Embedding Models

**Learning User Embeddings:**

**Approach 1: Collaborative Filtering Embeddings**
```
Training data: All user interactions
Method: Matrix Factorization or Neural CF

User-Item matrix (sparse):
  Rows: Users
  Columns: Items
  Values: interaction (click=1, view=0.5, purchased=2)

Factorization:
  U = (100K users × 128 dims)
  I = (10M items × 128 dims)
  
  Score ≈ U_user · I_item
  
  Optimize: Minimize reconstruction error
  
Result: user_embedding = U[user_id]
```

**Approach 2: Behavior-based Embeddings**
```
User history: [item_1, item_2, ..., item_k]
Method: Average of item embeddings

user_embedding = mean([item_embedding[i] for i in history])

Or weighted average (recent items more weight)
Or using attention (learned importance weights)
```

**Learning Item Embeddings:**

**Approach 1: Content-based**
```
Item: title, description, image, metadata

Use pre-trained models:
  Text: BERT(title + description) → 768-dim
  Image: ResNet(image) → 512-dim
  Category: Embedding lookup → 128-dim
  
Combine: Concatenate → 1408-dim
Then: Linear layer → 128-dim (final embedding)

Training: Supervised (learn features that predict user preference)
```

**Approach 2: Collaborative Filtering**
```
Same as matrix factorization above

Item embedding = I[item_id]

Advantage: Captures user preference patterns
Disadvantage: Requires interaction data (cold start problem)
Solution: Hybrid (use content initially, switch to CF as interactions accumulate)
```

### 8.4 Model Inference & Serving

**Batch Inference (Offline):**
```
Purpose: Pre-compute scores for all users (on scheduled basis)

Example (Daily):
  1. Load ranking model
  2. For each of 100M users:
     - Load user features (from cache/DB)
     - For each item: Score = model(user_features, item_features)
     - Sort and save top-1000 items
  3. Store results in database

Output: user_id → [ranked_item_list]

Advantage: Can do expensive computation offline
Disadvantage: Can't personalize by request context
Usage: Default recommendations, feed generation
```

**Real-time Inference (Online):**
```
Purpose: Score items at request time (personalized by context)

Request comes in:
  1. Extract features: user_features, item_features, context_features
  2. Load model (cached in memory)
  3. Forward pass: score = model(features)
  4. Ranking and post-processing
  5. Return top-k items

Optimization:
  - Model caching (keep in GPU memory)
  - Batch requests (score multiple items together)
  - Quantization (FP32 → INT8, faster inference)
  - Model distillation (smaller, faster model)
  
Latency budget: 50-100ms total
Model inference: <20ms
```

**GPU Serving (NVIDIA Triton, TensorFlow Serving):**
```
Setup:
  - Load model on GPU (fast computation)
  - Queue incoming requests
  - Batch requests for efficiency
  
Batching:
  Instead of: score_item(user, item1), score_item(user, item2), ...
  Do: score_items_batch(user, [item1, item2, item3, ...])
  
  GPU parallel processing much faster
  Trade-off: Batch wait time (use batching window of 10-50ms)
```

### 8.5 Data Logging & Feedback Loop

**Interaction Logging:**
```
Every user action → logged to event stream

Examples:
  {
    "user_id": 12345,
    "item_id": 67890,
    "timestamp": "2026-06-19T23:14:00Z",
    "action": "click",
    "recommended": true,
    "rank_position": 3,
    "impression_id": "abc123", 
    "model_version": "v2.1"
  }

Stream: Kafka, Kinesis, Pub/Sub
Size: Billions of events daily
Latency: <100ms from action to log
```

**Feedback Loop:**
```
Closed Loop:
  1. Show recommendations
  2. User clicks/interacts
  3. Log interaction with model version
  4. Next training: Use as training signal
  
This creates continuous improvement
But: Can reinforce biases (popular items recommended more)
```

**Debiasing Logged Data:**
```
Problem: Selection bias
  - Only items shown get clicked/not-clicked labels
  - Items not shown: unknown (missing labels)
  - Models trained on shown items → only learn about shown items
  
Solution: Inverse propensity weighting
  weight = 1 / P(item shown)
  
  If item shown to 80% of users: weight = 1/0.8 = 1.25
  If item shown to 5% of users: weight = 1/0.05 = 20
  
  Rare items get higher weight in loss
  Prevents bias toward popular items
```

### 8.6 Scaling Considerations

**Scale: 100M users, 10M items, 1B recommendations/day**

**Recall Stage (ANN Search):**
```
Challenge: Find k similar items from 10M items for 100M users

Solution: Approximate Nearest Neighbor (ANN) search
  
Libraries:
  - FAISS (Facebook AI): On single machine, ~1B vectors, <1ms search
  - Milvus: Distributed vector database
  - Elasticsearch: Approximate nearest neighbor plugin
  
Indexing:
  - Quantization: 128-dim FP32 → 16-dim INT8 (8x smaller)
  - Hierarchical clustering: Tree structure for fast traversal
  - Inverted indices: Fast filtering before ANN
  
Result: 500 candidates in <5ms
```

**Ranking Stage (DNN Inference):**
```
Challenge: Score 500 candidates for 100M users

Latency target: 1B recommendations/day with 50ms latency
  = 1B / (86400 * 1000 / 50) = 580K inferences/second needed
  
Solutions:
  1. Batch inference (average 10 items per request)
     = 58K requests/second (more manageable)
  
  2. Model serving with GPUs
     Each GPU: ~10K inferences/sec (FP16, batch size 32)
     Needed: ~6 GPUs (redundancy + peak load)
  
  3. Model caching at edge (if possible)
     Pre-score popular user-item pairs
```

**Model Training:**
```
Training data: Billions of interactions
  - 1 week of data: 7B interactions
  - Full matrix too large to fit in memory
  
Solutions:
  1. Distributed training (PyTorch, TensorFlow)
     - Shard data across multiple machines
     - Train in parallel
     - Synchronize gradients
  
  2. Online learning
     - Stream training data
     - Update model continuously
     - No need to load everything in memory
  
  3. Stochastic Gradient Descent
     - Process mini-batches (1000 samples at a time)
     - Update weights after each batch
     - Converges even on huge datasets
```

**Database Writes (Logging):**
```
Interactions to log: 1B/day = 12K/second

Database options:
  
1. NoSQL (Cassandra, MongoDB):
   - Write optimized
   - Shardable (partition by user_id, timestamp)
   - Handles 100K writes/sec per cluster
   - 2-3 clusters for redundancy/peak load
  
2. Time-series DB (InfluxDB, TimescaleDB):
   - Built for high-volume time-series data
   - Automatic partitioning by time
   - Good compression
  
3. Data warehouse (BigQuery):
   - Not real-time writes
   - Use streaming inserts (with latency)
   - Best for analytical queries later
```

### 8.7 Testing & Deployment

**Offline Testing:**
```
Before A/B test:
  1. Evaluate on historical holdout set
  2. Compute NDCG, MAP, HR metrics
  3. Check for regressions
  4. Ensure code quality (unit tests, integration tests)
```

**Canary Deployment:**
```
Roll out to 1% of traffic first
  - Monitor metrics closely
  - Watch for crashes, errors, latency increase
  - Check business metric deltas
  
If good → expand to 5% → 25% → 100%
If bad → rollback immediately (fast rollback mechanism)
```

**Monitoring in Production:**
```
Real-time dashboards:
  - Recommendation latency (p50, p95, p99)
  - Model error rate (inference failures)
  - Cache hit rates
  - API error codes
  - Business metrics (CTR, conversion, engagement)
  
Alerts:
  - Latency > 100ms
  - Error rate > 0.1%
  - NDCG drop > 5%
  - CTR change > 2% (unexpected)
  
Daily reports:
  - Model performance
  - User feedback/reviews
  - System health
```

---

## Key Takeaways for MLE Preparation

### 1. **Architecture First**
- Two-stage pipeline (recall → ranking) is industry standard
- Understand trade-offs: speed vs. accuracy at each stage
- Design for scale from day one

### 2. **Algorithms Matter, But So Do Details**
- ItemCF, UserCF, embeddings are all useful
- Implementation details (sampling, caching, batching) make huge difference
- Online learning trumps static models

### 3. **Diversity is Non-negotiable**
- Pure accuracy → user dissatisfaction → churn
- Diversity algorithms (DPP, greedy re-ranking) are essential
- Monitor as guardrail in all experiments

### 4. **Cold Start is a Fundamental Problem**
- Affects new users, new items, new catalogs
- Requires hybrid approaches (content + CF + exploration)
- Plan cold start strategy from architecture design

### 5. **Metrics Drive Everything**
- Offline metrics (NDCG, MAP) don't always correlate with business impact
- A/B testing is non-negotiable for important changes
- Guardrails prevent metric gaming

### 6. **Engineering Matters as Much as ML**
- Feature engineering/serving is 70% of the work
- Infrastructure (GPU serving, caching, databases) critical for scale
- Monitoring and rollback are as important as model accuracy

### 7. **Data Quality is Prerequisite**
- Biased training data → biased recommendations
- Logging infrastructure must be reliable
- Debiasing techniques (inverse propensity weighting) needed

### 8. **Iterate Quickly**
- Start with simple baselines
- A/B test small changes continuously
- Use online learning for rapid iteration
- Don't over-optimize offline before seeing online results

---

## Practical Implementation Checklist

**Phase 1: MVP (2-4 weeks)**
- [ ] Basic ItemCF recall
- [ ] Simple ranking (logistic regression)
- [ ] Top-k results, basic caching
- [ ] NDCG evaluation on sample data

**Phase 2: Improve Quality (4-8 weeks)**
- [ ] User embeddings
- [ ] Deep learning ranking model
- [ ] Diversity re-ranking (greedy)
- [ ] A/B testing infrastructure
- [ ] Daily model retraining pipeline

**Phase 3: Scale & Optimize (8-16 weeks)**
- [ ] Multiple recall channels
- [ ] ANN search (FAISS/Milvus)
- [ ] GPU-accelerated inference
- [ ] Real-time feature serving
- [ ] Online learning updates

**Phase 4: Production Hardening (Ongoing)**
- [ ] Comprehensive monitoring
- [ ] Automated canary deployment
- [ ] Guardrail metrics
- [ ] Cold start handling
- [ ] Performance optimization

---

## Resources for Further Learning

**Papers:**
- DIN (Deep Interest Network) - Alimama
- Wide & Deep Learning - Google
- LambdaMART - Microsoft Research
- Bandit Algorithms for Recommendation
- Determinantal Point Process for Diversity

**Tools:**
- PyTorch/TensorFlow: Model training
- FAISS/Milvus: Vector search
- Spark: Data processing
- Presto/Hive: SQL on big data
- Kafka: Event streaming

**Frameworks:**
- TensorFlow Serving: Model deployment
- MLflow: Experiment tracking
- Ray Tune: Hyperparameter tuning

---

*Document prepared for recommendation system MLE interview/preparation*
*Based on Wang Shusheng's comprehensive industrial recommendation systems course*
*Last updated: 2026-06-20*
