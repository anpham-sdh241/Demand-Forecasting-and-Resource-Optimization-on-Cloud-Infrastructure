# Lý Thuyết Deep Reinforcement Learning cho Bài Toán Allocation/Rescheduling

## 1. Giới Thiệu

Deep Reinforcement Learning (DRL) hay Học Tăng Cường Sâu là một phương pháp Machine Learning kết hợp giữa Reinforcement Learning và Deep Learning, cho phép một agent học cách thực hiện các hành động tối ưu trong một môi trường bằng cách tương tác trực tiếp với môi trường đó, sử dụng các mạng neural network sâu để học policy tối ưu.

**Tại sao Deep Reinforcement Learning phù hợp cho bài toán Allocation/Rescheduling?**

Trong bài toán phân bổ tài nguyên và lập lịch lại (rescheduling), DRL đã được chứng minh là hiệu quả trong việc tối ưu hóa việc sử dụng tài nguyên và cải thiện hiệu suất hệ thống. [[1]](https://link.springer.com/article/10.1007/s10462-025-11340-5) [[2]](https://arxiv.org/abs/2505.17359) DRL phù hợp với các bài toán này vì:

- **Sequential Decision Making**: Tối ưu hóa các quyết định theo thời gian, xem xét tác động dài hạn của mỗi quyết định
- **Xử lý Switching Cost**: Có thể học được patterns của chi phí chuyển đổi và tối ưu hóa tổng chi phí qua nhiều bước thời gian
- **Thích ứng với Uncertainty**: Xử lý được uncertainty và thay đổi nhu cầu động
- **Multi-objective Optimization**: Cân bằng nhiều mục tiêu mâu thuẫn thông qua reward function
- **Học từ Dữ Liệu**: Không cần mô hình toán học chính xác, học từ kinh nghiệm thực tế

## 2. Dạng Toán Học Tổng Quát

### 2.1. Markov Decision Process (MDP) Framework

Reinforcement Learning được mô hình hóa dưới dạng **Markov Decision Process (MDP)**, bao gồm 5 thành phần:

**MDP = (S, A, P, R, γ)**

Trong đó:
- **S**: State space (Không gian trạng thái) - Tập hợp tất cả các trạng thái có thể của môi trường
- **A**: Action space (Không gian hành động) - Tập hợp tất cả các hành động agent có thể thực hiện
- **P**: Transition probability (Xác suất chuyển trạng thái) - P(s'|s,a) là xác suất chuyển từ trạng thái s sang s' khi thực hiện hành động a
- **R**: Reward function (Hàm phần thưởng) - R(s,a,s') là phần thưởng nhận được khi chuyển từ s sang s' bằng hành động a
- **γ**: Discount factor (Hệ số chiết khấu) - γ ∈ [0,1], điều chỉnh tầm quan trọng của phần thưởng tương lai

### 2.2. Các Khái Niệm Cơ Bản

**Policy (Chính sách)**: π(a|s)
- Xác định xác suất chọn hành động a khi ở trạng thái s
- Mục tiêu là tìm policy tối ưu π* để tối đa hóa tổng phần thưởng kỳ vọng

**Value Function (Hàm giá trị)**:
- **State Value V^π(s)**: Giá trị kỳ vọng của tổng phần thưởng khi bắt đầu từ trạng thái s và theo policy π
  ```
  V^π(s) = E_π[Σ_{t=0}^∞ γ^t R_{t+1} | S_0 = s]
  ```
- **Action Value Q^π(s,a)**: Giá trị kỳ vọng khi thực hiện hành động a ở trạng thái s và sau đó theo policy π
  ```
  Q^π(s,a) = E_π[Σ_{t=0}^∞ γ^t R_{t+1} | S_0 = s, A_0 = a]
  ```

**Bellman Equation**:
- Phương trình đệ quy mô tả mối quan hệ giữa giá trị của trạng thái hiện tại và các trạng thái tiếp theo
- **Bellman Equation cho V^π**:
  ```
  V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γV^π(s')]
  ```
- **Bellman Equation cho Q^π**:
  ```
  Q^π(s,a) = Σ_{s'} P(s'|s,a) [R(s,a,s') + γ Σ_{a'} π(a'|s') Q^π(s',a')]
  ```

**Optimal Policy (Chính sách tối ưu)**: π*
- Policy tối ưu là policy có giá trị lớn nhất ở mọi trạng thái
- V^π*(s) ≥ V^π(s) với mọi s và mọi policy π

### 2.3. Mục Tiêu của Reinforcement Learning

Mục tiêu của RL là tìm policy tối ưu π* để tối đa hóa **expected return** (tổng phần thưởng kỳ vọng):

```
J(π) = E_π[Σ_{t=0}^∞ γ^t R_{t+1}]
```

Trong đó:
- R_t là phần thưởng tại thời điểm t
- γ là discount factor
- E_π là kỳ vọng theo policy π

## 3. Mô Hình Hóa Bài Toán Allocation/Rescheduling

### 3.1. Đặc Điểm của Bài Toán

Bài toán phân bổ tài nguyên và lập lịch lại có các đặc điểm chính:

1. **Sequential Decision Making**: 
   - Quyết định tại mỗi timestep phụ thuộc vào trạng thái hiện tại
   - Ảnh hưởng đến các quyết định tương lai
   - Cần tối ưu hóa tổng chi phí qua nhiều bước thời gian

2. **Long-term Optimization**:
   - Agent cần học cách cân bằng giữa chi phí ngắn hạn và dài hạn
   - Switching cost tạo ra dependencies giữa các quyết định
   - Discount factor γ điều chỉnh tầm quan trọng của tương lai

3. **Multi-objective**:
   - Cần cân bằng nhiều mục tiêu mâu thuẫn:
     - Resource availability vs Cost
     - Switching cost vs Operational cost
     - Efficiency vs Safety margin
   - Reward function cho phép điều chỉnh trade-off này

4. **Uncertainty**:
   - Nhu cầu tài nguyên không chắc chắn
   - Forecast có thể không chính xác
   - Agent cần học cách xử lý uncertainty này

5. **State Space Complexity**:
   - State space lớn và phức tạp
   - Cần deep neural networks để học representation hiệu quả
   - Có thể bao gồm: current demand, forecast, current allocation, time features

### 3.2. State Space (Không Gian Trạng Thái)

Trong bài toán Allocation/Rescheduling, state thường bao gồm:

**1. Current Demand (Nhu cầu hiện tại)**:
- Nhu cầu tài nguyên hiện tại
- Overflow (nếu có) khi nhu cầu vượt quá capacity

**2. Forecast Information (Thông tin dự báo)**:
- Dự báo nhu cầu cho các bước thời gian tiếp theo
- Giúp agent nhìn xa về tương lai để đưa ra quyết định tốt hơn

**3. Current Allocation (Phân bổ hiện tại)**:
- Trạng thái phân bổ tài nguyên hiện tại
- Quan trọng để tính switching cost khi thay đổi allocation

**4. Time Features (Đặc trưng thời gian)**:
- Thông tin về thời gian (giờ, ngày, tuần)
- Giúp agent học được temporal patterns (daily/weekly cycles)

### 3.3. Action Space (Không Gian Hành Động)

Action trong bài toán Allocation/Rescheduling thường là:

- **Discrete actions**: Số lượng tài nguyên cần phân bổ cho từng loại
- **MultiDiscrete space**: Mỗi thành phần là discrete và độc lập
- Agent quyết định allocation mới dựa trên state hiện tại

**Rescheduling Actions**:
- Agent có thể quyết định giữ nguyên allocation để tránh switching cost
- Hoặc thay đổi allocation để đáp ứng nhu cầu mới hoặc giảm chi phí
- Quyết định này phụ thuộc vào trade-off giữa switching cost và operational cost

### 3.4. Reward Function (Hàm Phần Thưởng)

Reward function trong bài toán Allocation/Rescheduling thường bao gồm:

**1. Performance Metrics**:
- Phần thưởng/phạt dựa trên việc đáp ứng nhu cầu
- Penalty khi vi phạm SLA hoặc không đáp ứng đủ tài nguyên

**2. Cost Components**:
- Operational cost: Chi phí vận hành tài nguyên
- Switching cost: Chi phí chuyển đổi giữa các cấu hình
- Tối thiểu hóa tổng chi phí

**3. Efficiency Metrics**:
- Utilization: Sử dụng hiệu quả tài nguyên đã allocate
- Right-sizing: Phân bổ đúng lượng cần thiết

**Reward Shaping**:
- Thiết kế reward function là một nghệ thuật, ảnh hưởng lớn đến behavior của agent
- Cần cân bằng các thành phần để đạt được mục tiêu mong muốn
- Reward shaping không đúng có thể dẫn đến behavior không mong muốn

## 4. Phương Pháp Giải: Proximal Policy Optimization (PPO)

### 4.1. Giới Thiệu về PPO

**Proximal Policy Optimization (PPO)** là thuật toán policy gradient được đề xuất bởi OpenAI vào năm 2017. PPO là một cải tiến của Trust Region Policy Optimization (TRPO), đơn giản hơn để triển khai nhưng vẫn giữ được tính ổn định trong quá trình training. [[3]](https://arxiv.org/abs/1707.06347)

**Tại sao chọn PPO?**
- **Stable Training**: Tránh được policy update quá lớn gây mất ổn định
- **Sample Efficient**: Sử dụng dữ liệu hiệu quả hơn so với các phương pháp on-policy khác
- **Easy to Implement**: Đơn giản hơn TRPO nhưng vẫn hiệu quả
- **Widely Used**: Được sử dụng rộng rãi và có nhiều implementation sẵn có

### 4.2. Nguyên Lý Cơ Bản

PPO thuộc nhóm **Policy Gradient** methods, tối ưu hóa policy trực tiếp bằng cách:

1. Thu thập samples từ policy hiện tại
2. Tính toán advantage estimates
3. Cập nhật policy để tăng xác suất các hành động tốt và giảm xác suất các hành động xấu

**Policy Gradient Objective**:
```
L^PG(θ) = E_t [log π_θ(a_t|s_t) × A_t]
```

Trong đó:
- π_θ(a_t|s_t) là xác suất chọn hành động a_t ở trạng thái s_t theo policy θ
- A_t là advantage estimate (độ lợi của hành động so với baseline)

### 4.3. Clipped Objective của PPO

PPO sử dụng **clipped objective** để tránh policy update quá lớn:

```
L^CLIP(θ) = E_t [min(
    r_t(θ) × A_t,
    clip(r_t(θ), 1-ε, 1+ε) × A_t
)]
```

Trong đó:
- **r_t(θ)**: Importance sampling ratio = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
- **ε**: Clip range (thường là 0.1 hoặc 0.2)
- **A_t**: Advantage estimate

**Cơ chế hoạt động**:
- Nếu advantage A_t > 0 (hành động tốt): Tăng xác suất hành động, nhưng giới hạn bởi (1+ε)
- Nếu advantage A_t < 0 (hành động xấu): Giảm xác suất hành động, nhưng giới hạn bởi (1-ε)
- Điều này ngăn policy thay đổi quá nhanh, giữ training ổn định

### 4.4. Advantage Estimation

PPO sử dụng **Generalized Advantage Estimation (GAE)** để ước lượng advantage:

```
A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
```

Trong đó:
- **δ_t**: TD error = r_t + γV(s_{t+1}) - V(s_t)
- **γ**: Discount factor
- **λ**: GAE parameter (thường là 0.95)

GAE kết hợp giữa Monte Carlo (λ=1) và TD learning (λ=0), giảm variance trong khi vẫn giữ bias thấp.

### 4.5. Thuật Toán PPO

**Input**: 
- Initial policy parameters θ₀
- Initial value function parameters φ₀

**For iteration = 1, 2, ... do**:
1. **Collect trajectories**: Chạy policy π_θ trong môi trường để thu thập N steps
2. **Compute advantages**: Tính A_t sử dụng GAE
3. **Update policy**: Tối ưu L^CLIP(θ) với K epochs và minibatch size M
4. **Update value function**: Tối ưu value function để fit với returns

**Output**: Policy tối ưu π_θ*

### 4.6. Hyperparameters Quan Trọng

**Learning Rate**: Tốc độ học (thường 3e-4)
- Quá cao: Training không ổn định
- Quá thấp: Học chậm

**Clip Range (ε)**: Phạm vi clip (thường 0.1-0.2)
- Kiểm soát độ lớn của policy update

**n_steps**: Số steps thu thập trước mỗi update (thường 2048)
- Trade-off giữa sample efficiency và update frequency

**n_epochs**: Số epochs tối ưu trên cùng một batch (thường 4-10)
- Tận dụng dữ liệu hiệu quả hơn

**Batch Size**: Kích thước minibatch (thường 64-256)
- Ảnh hưởng đến stability và speed

**Gamma (γ)**: Discount factor (thường 0.99)
- Điều chỉnh tầm quan trọng của phần thưởng tương lai
- Quan trọng cho bài toán rescheduling với switching cost

**GAE Lambda (λ)**: GAE parameter (thường 0.95)
- Cân bằng giữa bias và variance trong advantage estimation

**Entropy Coefficient**: Khuyến khích exploration (thường 0.01)
- Giúp agent khám phá các hành động mới
- Quan trọng để tránh stuck ở local optima

### 4.7. Architecture của Neural Network

Trong PPO, thường sử dụng **Actor-Critic** architecture:

**Actor Network (Policy Network)**:
- Input: State s
- Output: Policy distribution π(a|s)
- Mục tiêu: Tối ưu policy để tối đa hóa expected return

**Critic Network (Value Network)**:
- Input: State s
- Output: Value estimate V(s)
- Mục tiêu: Ước lượng giá trị của trạng thái để tính advantage

**Shared Network**:
- Thường chia sẻ các layers đầu giữa Actor và Critic
- Giảm số lượng parameters và tăng hiệu quả training

## 5. Ưu Điểm và Hạn Chế

Deep Reinforcement Learning là một phương pháp mạnh mẽ cho các bài toán tối ưu hóa phức tạp, đặc biệt là các bài toán có tính sequential và uncertainty như Allocation/Rescheduling. Tuy nhiên, để áp dụng hiệu quả, cần hiểu rõ cả những ưu điểm và hạn chế của phương pháp này.

### 5.1. Ưu Điểm

Mặc dù có một số hạn chế, Deep Reinforcement Learning vẫn là một phương pháp được sử dụng rộng rãi cho bài toán Allocation/Rescheduling nhờ những ưu điểm nổi bật sau đây:

1. **Xử lý Sequential Decisions hiệu quả**:
   - Tối ưu hóa các quyết định theo thời gian, xem xét tác động dài hạn
   - Có thể học được switching cost patterns và tối ưu chi phí chuyển đổi qua nhiều bước
   - Phù hợp với bài toán rescheduling cần nhìn xa về tương lai
   - Agent học được khi nào nên thay đổi allocation và khi nào nên giữ nguyên

2. **Tối ưu Multi-objective tự nhiên**:
   - Có thể cân bằng nhiều mục tiêu mâu thuẫn thông qua reward function
   - Học được trade-off giữa các mục tiêu từ dữ liệu
   - Linh hoạt trong việc điều chỉnh weights để phù hợp với từng scenario
   - Agent học được cách cân bằng các mục tiêu một cách tự nhiên

3. **Thích ứng với Uncertainty**:
   - Có thể học cách xử lý uncertainty trong nhu cầu và forecast
   - Robust với các thay đổi trong môi trường
   - Agent học được cách đối phó với forecast errors
   - Thích ứng với dynamic environments có patterns thay đổi

4. **Học Patterns phức tạp**:
   - Deep neural networks có khả năng học các patterns phức tạp từ dữ liệu
   - Có thể học được các quan hệ phi tuyến
   - Xử lý được state space lớn và phức tạp
   - Học được temporal patterns (daily/weekly cycles) từ time features

5. **Không cần nghiệm nguyên**:
   - Action space có thể là discrete (số nguyên)
   - Phù hợp với bài toán thực tế cần nghiệm nguyên
   - Không có vấn đề về divisibility

6. **Không cần mô hình toán học chính xác**:
   - Agent học từ kinh nghiệm thông qua tương tác với môi trường
   - Không cần biết chính xác transition probabilities
   - Phù hợp với các bài toán phức tạp khó mô hình hóa bằng toán học

### 5.2. Hạn Chế

Mặc dù có nhiều ưu điểm, Deep Reinforcement Learning cũng có những hạn chế nhất định do bản chất của phương pháp học từ dữ liệu. Việc hiểu rõ các hạn chế này giúp xác định khi nào DRL phù hợp và khi nào cần các phương pháp thay thế:

1. **Cần nhiều dữ liệu và thời gian training**:
   - Agent cần tương tác với môi trường nhiều lần để học (hàng triệu timesteps)
   - Training có thể mất hàng giờ đến hàng ngày tùy vào độ phức tạp
   - Sample efficiency thấp hơn so với supervised learning
   - Yêu cầu computational resources đáng kể (GPU)

2. **Không đảm bảo optimality**:
   - DRL tìm được nghiệm gần tối ưu, không đảm bảo global optimum
   - Phụ thuộc vào initialization và hyperparameters
   - Có thể bị stuck ở local optima
   - Performance có thể không nhất quán giữa các lần training

3. **Hyperparameter tuning phức tạp**:
   - Có nhiều hyperparameters cần điều chỉnh (learning rate, clip range, network architecture, etc.)
   - Reward function design cũng là một thách thức
   - Ảnh hưởng lớn đến performance nhưng khó tune
   - Cần nhiều thử nghiệm và kinh nghiệm

4. **Khó interpret và debug**:
   - Khó hiểu tại sao agent đưa ra quyết định cụ thể
   - Black box nature của deep neural networks
   - Khó debug khi performance không tốt
   - Khó giải thích cho stakeholders

5. **Reward function design**:
   - Thiết kế reward function là một nghệ thuật, ảnh hưởng lớn đến behavior của agent
   - Reward shaping không đúng có thể dẫn đến behavior không mong muốn
   - Cần cân bằng các thành phần trong reward function
   - Agent có thể học cách exploit reward thay vì giải quyết bài toán thực tế

6. **Stability và reproducibility**:
   - Training có thể không ổn định, đặc biệt với learning rate cao
   - Kết quả có thể khác nhau giữa các lần chạy do randomness
   - Cần set random seeds để reproducibility
   - Có thể cần nhiều lần training để có kết quả tốt

7. **Exploration vs Exploitation trade-off**:
   - Agent cần khám phá môi trường nhưng cũng cần khai thác những gì đã học
   - Cân bằng này khó và phụ thuộc vào entropy coefficient
   - Quá ít exploration: có thể bỏ lỡ nghiệm tốt hơn
   - Quá nhiều exploration: học chậm và không hiệu quả

8. **Computational resources**:
   - Cần GPU để training hiệu quả với deep neural networks
   - Memory requirements cao cho large state/action spaces
   - Training cost có thể cao

## Tài Liệu Tham Khảo

1. **Hady, M.A., Hu, S., Pratama, M., et al. (2025).** "Multi-agent reinforcement learning for resources allocation optimization: a survey." *Artificial Intelligence Review*, 58, 354.  
   https://link.springer.com/article/10.1007/s10462-025-11340-5  
   *(Survey về MARL trong Resource Allocation Optimization)*

2. **Ding, X., Zhang, Y., Chen, B., et al. (2025).** "Towards VM Rescheduling Optimization Through Deep Reinforcement Learning." *arXiv preprint arXiv:2505.17359*.  
   https://arxiv.org/abs/2505.17359  
   *(Paper về VM Rescheduling với DRL)*

3. **Schulman, J., et al. (2017).** "Proximal Policy Optimization Algorithms." *arXiv preprint arXiv:1707.06347*.  
   *(Paper gốc về PPO algorithm)*

4. **Sutton, R. S., & Barto, A. G. (2018).** *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.  
   *(Giáo trình cơ bản về Reinforcement Learning và MDP framework)*

5. **Stable-Baselines3 Documentation**  
   https://stable-baselines3.readthedocs.io/  
   *(Thư viện RL có implementation PPO)*

6. **OpenAI Spinning Up - PPO**  
   https://spinningup.openai.com/en/latest/algorithms/ppo.html  
   *(Giải thích chi tiết về PPO với code examples)*

