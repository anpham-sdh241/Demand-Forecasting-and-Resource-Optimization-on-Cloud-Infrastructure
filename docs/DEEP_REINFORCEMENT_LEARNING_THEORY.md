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

$$\text{MDP} = (S, A, P, R, \gamma)$$

Trong đó:
- **S**: State space (Không gian trạng thái) - Tập hợp tất cả các trạng thái có thể của môi trường
- **A**: Action space (Không gian hành động) - Tập hợp tất cả các hành động agent có thể thực hiện
- **P**: Transition probability (Xác suất chuyển trạng thái) - $P(s'|s,a)$ là xác suất chuyển từ trạng thái $s$ sang $s'$ khi thực hiện hành động $a$
- **R**: Reward function (Hàm phần thưởng) - $R(s,a,s')$ là phần thưởng nhận được khi chuyển từ $s$ sang $s'$ bằng hành động $a$
- **γ**: Discount factor (Hệ số chiết khấu) - $\gamma \in [0,1]$, điều chỉnh tầm quan trọng của phần thưởng tương lai

### 2.2. Các Khái Niệm Cơ Bản

**Policy (Chính sách)**: $\pi(a|s)$
- Xác định xác suất chọn hành động $a$ khi ở trạng thái $s$
- Mục tiêu là tìm policy tối ưu $\pi^*$ để tối đa hóa tổng phần thưởng kỳ vọng

**Value Function (Hàm giá trị)**:
- **State Value $V^{\pi}(s)$**: Giá trị kỳ vọng của tổng phần thưởng khi bắt đầu từ trạng thái $s$ và theo policy $\pi$
  $$V^{\pi}(s) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t R_{t+1} \mid S_0 = s\right]$$
- **Action Value $Q^{\pi}(s,a)$**: Giá trị kỳ vọng khi thực hiện hành động $a$ ở trạng thái $s$ và sau đó theo policy $\pi$
  $$Q^{\pi}(s,a) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t R_{t+1} \mid S_0 = s, A_0 = a\right]$$

**Bellman Equation**:
- Phương trình đệ quy mô tả mối quan hệ giữa giá trị của trạng thái hiện tại và các trạng thái tiếp theo
- **Bellman Equation cho $V^{\pi}$**:
  $$V^{\pi}(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[R(s,a,s') + \gamma V^{\pi}(s')\right]$$
- **Bellman Equation cho $Q^{\pi}$**:
  $$Q^{\pi}(s,a) = \sum_{s'} P(s'|s,a) \left[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^{\pi}(s',a')\right]$$

**Optimal Policy (Chính sách tối ưu)**: $\pi^*$
- Policy tối ưu là policy có giá trị lớn nhất ở mọi trạng thái
- $V^{\pi^*}(s) \geq V^{\pi}(s)$ với mọi $s$ và mọi policy $\pi$

### 2.3. Mục Tiêu của Reinforcement Learning

Mục tiêu của RL là tìm policy tối ưu $\pi^*$ để tối đa hóa **expected return** (tổng phần thưởng kỳ vọng):

$$J(\pi) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t R_{t+1}\right]$$

Trong đó:
- $R_t$ là phần thưởng tại thời điểm $t$
- $\gamma$ là discount factor
- $\mathbb{E}_{\pi}$ là kỳ vọng theo policy $\pi$

## 3. Mô Hình Hóa Bài Toán Allocation/Rescheduling

### 3.1. Đặc Điểm của Bài Toán

Bài toán phân bổ tài nguyên và lập lịch lại được mô hình hóa trong framework MDP với các đặc điểm chính:

- **Sequential Decision Making**: Quyết định tại mỗi timestep phụ thuộc vào trạng thái hiện tại và ảnh hưởng đến các quyết định tương lai, yêu cầu tối ưu hóa tổng chi phí qua nhiều bước thời gian.

- **Long-term Optimization**: Agent cần cân bằng giữa chi phí ngắn hạn và dài hạn, đặc biệt khi switching cost tạo ra dependencies giữa các quyết định.

- **Multi-objective**: Cần cân bằng nhiều mục tiêu mâu thuẫn (resource availability vs cost, switching cost vs operational cost) thông qua reward function.

- **Uncertainty**: Nhu cầu tài nguyên và forecast không chắc chắn, agent cần học cách xử lý uncertainty này.

### 3.2. State Space (Không Gian Trạng Thái)

State trong bài toán Allocation/Rescheduling bao gồm:

- **Current Demand**: Nhu cầu tài nguyên hiện tại và overflow (nếu có)
- **Forecast Information**: Dự báo nhu cầu cho các bước thời gian tiếp theo
- **Current Allocation**: Trạng thái phân bổ hiện tại (quan trọng để tính switching cost)
- **Time Features**: Thông tin thời gian (giờ, ngày, tuần) để học temporal patterns

### 3.3. Action Space (Không Gian Hành Động)

Action space là **MultiDiscrete**, mỗi action là một vector xác định số lượng tài nguyên phân bổ cho từng loại. Agent có thể:
- Giữ nguyên allocation để tránh switching cost
- Thay đổi allocation để đáp ứng nhu cầu mới hoặc giảm chi phí

Quyết định phụ thuộc vào trade-off giữa switching cost và operational cost.

### 3.4. Reward Function (Hàm Phần Thưởng)

Reward function bao gồm ba thành phần chính:

- **Performance Metrics**: Phần thưởng/phạt dựa trên việc đáp ứng nhu cầu, penalty khi vi phạm SLA
- **Cost Components**: Operational cost (chi phí vận hành) và switching cost (chi phí chuyển đổi)
- **Efficiency Metrics**: Utilization và right-sizing của tài nguyên

Thiết kế reward function là yếu tố quan trọng, cần cân bằng các thành phần để đạt được mục tiêu mong muốn.

## 4. Phương Pháp Giải: Proximal Policy Optimization (PPO)

### 4.1. Giới Thiệu về PPO

**Proximal Policy Optimization (PPO)** là thuật toán policy gradient được đề xuất bởi OpenAI vào năm 2017, là cải tiến của TRPO với cách triển khai đơn giản hơn nhưng vẫn giữ được tính ổn định. [[3]](https://arxiv.org/abs/1707.06347) PPO được chọn vì:
- **Stable Training**: Tránh policy update quá lớn gây mất ổn định
- **Sample Efficient**: Sử dụng dữ liệu hiệu quả hơn các phương pháp on-policy khác
- **Easy to Implement**: Đơn giản hơn TRPO nhưng vẫn hiệu quả

### 4.2. Nguyên Lý Cơ Bản

PPO thuộc nhóm **Policy Gradient** methods, tối ưu hóa policy trực tiếp bằng cách:
1. Thu thập samples từ policy hiện tại
2. Tính toán advantage estimates
3. Cập nhật policy để tăng xác suất các hành động tốt và giảm xác suất các hành động xấu

**Policy Gradient Objective**:
$$L^{PG}(\theta) = \mathbb{E}_t \left[\log \pi_{\theta}(a_t|s_t) \times A_t\right]$$

Trong đó $\pi_{\theta}(a_t|s_t)$ là xác suất chọn hành động $a_t$ ở trạng thái $s_t$, và $A_t$ là advantage estimate.

### 4.3. Clipped Objective của PPO

PPO sử dụng **clipped objective** để tránh policy update quá lớn:

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[\min\left(
    r_t(\theta) \times A_t,
    \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \times A_t
\right)\right]$$

Trong đó $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ là importance sampling ratio, và $\epsilon$ là clip range (thường 0.1-0.2). Cơ chế này giới hạn độ lớn của policy update, giữ training ổn định.

### 4.4. Advantage Estimation

PPO sử dụng **Generalized Advantage Estimation (GAE)** để ước lượng advantage:

$$A_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots$$

Trong đó $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ là TD error, $\gamma$ là discount factor, và $\lambda$ là GAE parameter (thường 0.95). GAE kết hợp Monte Carlo và TD learning, giảm variance trong khi vẫn giữ bias thấp.

### 4.5. Thuật Toán PPO

Thuật toán PPO hoạt động theo chu kỳ lặp lại, mỗi vòng lặp (iteration) bao gồm 4 bước chính:

**Input**: Khởi tạo các tham số ban đầu cho Actor network (mạng chính sách) và Critic network (mạng đánh giá giá trị).

**For iteration = 1, 2, ... do**:

**Collect trajectories (Thu thập dữ liệu)**: Agent sử dụng chính sách hiện tại để tương tác với môi trường và thu thập một số lượng bước nhất định (thường là 2048 bước). Mỗi bước ghi lại thông tin: trạng thái hiện tại, hành động đã thực hiện, phần thưởng nhận được, và trạng thái tiếp theo. Tập dữ liệu này sẽ được sử dụng để cải thiện chính sách trong các bước tiếp theo.

**Compute advantages (Tính toán advantage)**: Sử dụng Critic network để ước lượng giá trị của mỗi trạng thái. Sau đó tính toán advantage - một chỉ số cho biết hành động tại trạng thái đó tốt hơn hay tệ hơn so với mức trung bình. Advantage được tính bằng phương pháp GAE (Generalized Advantage Estimation), kết hợp thông tin từ nhiều bước thời gian để có ước lượng chính xác hơn. Advantage dương nghĩa là hành động tốt, advantage âm nghĩa là hành động tệ.

**Update policy (Cập nhật policy)**: Cập nhật Actor network (mạng chính sách) dựa trên advantage đã tính. Quá trình này được thực hiện qua nhiều epochs (thường 4-10 lần) trên cùng một tập dữ liệu để tận dụng tối đa thông tin. Dữ liệu được chia thành các batch nhỏ để tối ưu hóa hiệu quả. Mục tiêu là tăng xác suất chọn các hành động có advantage cao (hành động tốt) và giảm xác suất chọn các hành động có advantage thấp (hành động tệ). PPO sử dụng cơ chế clipping để đảm bảo chính sách không thay đổi quá nhanh, giữ cho quá trình training ổn định.

**Update value function (Cập nhật value function)**: Cập nhật Critic network (mạng đánh giá giá trị) để dự đoán chính xác hơn giá trị của các trạng thái. Mạng này được huấn luyện để dự đoán tổng phần thưởng kỳ vọng từ mỗi trạng thái, giúp tính toán advantage chính xác hơn trong các vòng lặp tiếp theo.

**Output**: Chính sách tối ưu sau khi quá trình training hội tụ hoặc đạt được hiệu suất mong muốn.

Quá trình này lặp lại cho đến khi chính sách học được cách đưa ra quyết định tốt. Mỗi vòng lặp sử dụng dữ liệu mới được thu thập từ chính sách hiện tại, đảm bảo agent luôn học từ những gì nó đang làm (on-policy learning).

### 4.6. Hyperparameters Quan Trọng

Các hyperparameters chính của PPO:
- **Learning Rate**: Tốc độ học (thường $3 \times 10^{-4}$)
- **Clip Range ($\epsilon$)**: Phạm vi clip (thường 0.1-0.2)
- **n_steps**: Số steps thu thập trước mỗi update (thường 2048)
- **n_epochs**: Số epochs tối ưu trên cùng một batch (thường 4-10)
- **Gamma ($\gamma$)**: Discount factor (thường 0.99), quan trọng cho bài toán rescheduling
- **GAE Lambda ($\lambda$)**: GAE parameter (thường 0.95)
- **Entropy Coefficient**: Khuyến khích exploration (thường 0.01)

### 4.7. Architecture của Neural Network

PPO sử dụng **Actor-Critic** architecture:
- **Actor Network**: Nhận state $s$, xuất policy distribution $\pi(a|s)$
- **Critic Network**: Nhận state $s$, xuất value estimate $V(s)$
- **Shared Network**: Thường chia sẻ các layers đầu giữa Actor và Critic để giảm parameters và tăng hiệu quả training

## 5. Ưu Điểm và Hạn Chế

Deep Reinforcement Learning là một phương pháp mạnh mẽ cho các bài toán tối ưu hóa phức tạp, đặc biệt là các bài toán có tính sequential và uncertainty như Allocation/Rescheduling. Tuy nhiên, để áp dụng hiệu quả, cần hiểu rõ cả những ưu điểm và hạn chế của phương pháp này.

### 5.1. Ưu Điểm

Deep Reinforcement Learning được sử dụng rộng rãi cho bài toán Allocation/Rescheduling nhờ những ưu điểm sau:

**Xử lý Sequential Decisions hiệu quả**: DRL tối ưu hóa các quyết định theo thời gian, xem xét tác động dài hạn. Agent học được patterns của switching cost và tối ưu chi phí chuyển đổi qua nhiều bước, phù hợp với bài toán rescheduling cần nhìn xa về tương lai.

**Tối ưu Multi-objective tự nhiên**: DRL cân bằng nhiều mục tiêu mâu thuẫn thông qua reward function. Agent học được trade-off giữa các mục tiêu từ dữ liệu một cách tự nhiên, linh hoạt điều chỉnh để phù hợp với từng scenario.

**Thích ứng với Uncertainty**: DRL học cách xử lý uncertainty trong nhu cầu và forecast, robust với các thay đổi trong môi trường và thích ứng với dynamic environments có patterns thay đổi.

**Học Patterns phức tạp**: Deep neural networks học các patterns phức tạp và quan hệ phi tuyến từ dữ liệu, xử lý được state space lớn và học được temporal patterns như daily/weekly cycles.

**Không cần nghiệm nguyên**: Action space có thể là discrete (số nguyên), phù hợp với bài toán thực tế cần nghiệm nguyên như phân bổ số lượng VM.

**Không cần mô hình toán học chính xác**: Agent học từ kinh nghiệm thông qua tương tác với môi trường, không cần biết chính xác transition probabilities, phù hợp với các bài toán phức tạp khó mô hình hóa bằng toán học.

### 5.2. Hạn Chế

Deep Reinforcement Learning có những hạn chế nhất định do bản chất của phương pháp học từ dữ liệu:

**Cần nhiều dữ liệu và thời gian training**: Agent cần tương tác với môi trường nhiều lần (hàng triệu timesteps), training có thể mất hàng giờ đến hàng ngày. Sample efficiency thấp hơn supervised learning và yêu cầu GPU để training hiệu quả.

**Không đảm bảo optimality**: DRL tìm được nghiệm gần tối ưu nhưng không đảm bảo global optimum. Kết quả phụ thuộc vào initialization và hyperparameters, có thể bị stuck ở local optima và performance không nhất quán giữa các lần training.

**Hyperparameter tuning phức tạp**: Có nhiều hyperparameters cần điều chỉnh (learning rate, clip range, network architecture, reward function design) ảnh hưởng lớn đến performance nhưng khó tune, đòi hỏi nhiều thử nghiệm.

**Khó interpret và debug**: Khó hiểu tại sao agent đưa ra quyết định cụ thể do black box nature của deep neural networks. Debug khó khăn khi performance không tốt và khó giải thích cho stakeholders.

**Reward function design**: Thiết kế reward function là thách thức lớn, ảnh hưởng trực tiếp đến behavior của agent. Reward shaping không đúng có thể dẫn đến behavior không mong muốn, và agent có thể exploit reward thay vì giải quyết bài toán thực tế.

**Stability và reproducibility**: Training có thể không ổn định, kết quả khác nhau giữa các lần chạy do randomness. Có thể cần nhiều lần training để có kết quả tốt và ổn định.

**Exploration vs Exploitation trade-off**: Cân bằng giữa khám phá môi trường và khai thác những gì đã học là khó, phụ thuộc vào entropy coefficient. Quá ít exploration bỏ lỡ nghiệm tốt hơn, quá nhiều exploration làm chậm quá trình học.

**Computational resources**: Cần GPU để training hiệu quả, memory requirements cao cho large state/action spaces, và training cost có thể cao.

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

