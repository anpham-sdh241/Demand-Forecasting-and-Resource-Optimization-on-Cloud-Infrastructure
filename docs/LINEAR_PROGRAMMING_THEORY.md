# Lý Thuyết Linear Programming (Quy Hoạch Tuyến Tính) — Báo Cáo Chi Tiết

## 1. Phạm Vi & Mục Tiêu
- Trình bày đầy đủ khung lý thuyết Linear Programming (LP) cho người đọc kỹ thuật.
- Làm rõ mô hình toán học, tính chất hình học, phương pháp giải, duality, phân tích độ nhạy, ưu/nhược điểm và các biến thể mở rộng.
- Không đi sâu vào lịch sử hay ứng dụng cụ thể; tập trung vào nền tảng lý thuyết và công thức.

## 2. Dạng Toán Học Tổng Quát

### 2.1. Dạng Chuẩn (Standard Form)

Một bài toán Linear Programming dạng chuẩn có dạng:

**Minimize (hoặc Maximize):**
```
Z = c₁x₁ + c₂x₂ + ... + cₙxₙ = c'x
```

**Subject to (Ràng buộc):**
```
a₁₁x₁ + a₁₂x₂ + ... + a₁ₙxₙ  {≤, =, ≥}  b₁
a₂₁x₁ + a₂₂x₂ + ... + a₂ₙxₙ  {≤, =, ≥}  b₂
...
aₘ₁x₁ + aₘ₂x₂ + ... + aₘₙxₙ  {≤, =, ≥}  bₘ

x₁, x₂, ..., xₙ ≥ 0  (non-negativity constraints)
```

**Dạng ma trận:**
```
Minimize:    Z = c'x
Subject to:  Ax {≤, =, ≥} b
             x ≥ 0
```

Trong đó:
- **x** = [x₁, x₂, ..., xₙ]ᵀ ∈ ℝⁿ: Vector biến quyết định (decision variables)
- **c** = [c₁, c₂, ..., cₙ]ᵀ ∈ ℝⁿ: Vector hệ số mục tiêu (objective coefficients)
- **A** ∈ ℝᵐˣⁿ: Ma trận hệ số ràng buộc (constraint matrix)
- **b** = [b₁, b₂, ..., bₘ]ᵀ ∈ ℝᵐ: Vector vế phải (right-hand side vector)

### 2.2. Các Dạng Ràng Buộc

**1. Inequality Constraints (Bất phương trình):**
```
Ax ≤ b   (less than or equal)
Ax ≥ b   (greater than or equal)
```

**2. Equality Constraints (Phương trình):**
```
Ax = b
```

**3. Bounds (Giới hạn biến):**
```
lᵢ ≤ xᵢ ≤ uᵢ   (bounded variable)
xᵢ ≥ 0         (non-negative variable)
xᵢ ∈ ℝ         (free variable, unrestricted)
```

### 2.3. Chuyển Đổi Về Dạng Chuẩn

Mọi bài toán LP đều có thể chuyển về dạng chuẩn:

**Minimize: c'x**  
**Subject to: Ax = b, x ≥ 0**

**Các phép chuyển đổi:**

1. **Maximize → Minimize:**  
   Maximize Z ⟺ Minimize -Z

2. **Inequality → Equality (thêm slack/surplus variables):**
   - Σaᵢⱼxⱼ ≤ bᵢ ⟺ Σaᵢⱼxⱼ + sᵢ = bᵢ, sᵢ ≥ 0 (slack)
   - Σaᵢⱼxⱼ ≥ bᵢ ⟺ Σaᵢⱼxⱼ - sᵢ = bᵢ, sᵢ ≥ 0 (surplus)

3. **Free variable → Non-negative variables:**  
   xⱼ ∈ ℝ ⟺ xⱼ = xⱼ⁺ - xⱼ⁻, xⱼ⁺, xⱼ⁻ ≥ 0

4. **Bounded variable → Constraints:**  
   lⱼ ≤ xⱼ ≤ uⱼ ⟺ xⱼ ≥ lⱼ và xⱼ ≤ uⱼ

## 3. Khái Niệm Cơ Bản

### 3.1. Feasible Region (Miền Chấp Nhận Được)
- Tập hợp tất cả các điểm x thỏa mãn tất cả ràng buộc
- Là một tập lồi (convex set) trong không gian ℝⁿ
- Có thể là: bounded (bị chặn), unbounded (không bị chặn), hoặc empty (rỗng - infeasible)

### 3.2. Optimal Solution (Nghiệm Tối Ưu)
- Một điểm trong feasible region có giá trị hàm mục tiêu tốt nhất (min hoặc max)
- Có thể có **unique optimal** (duy nhất) hoặc **multiple optimal solutions** (nhiều nghiệm)
- Nếu feasible region là bounded và non-empty, nghiệm tối ưu luôn tồn tại

### 3.3. Extreme Points (Điểm Cực Trị)
- Còn gọi là **vertices** (đỉnh) hoặc **corner points**
- Là các điểm không thể biểu diễn như tổ hợp lồi của 2 điểm khác trong feasible region
- **Định lý cơ bản**: Nếu LP có nghiệm tối ưu, tồn tại ít nhất một extreme point là nghiệm tối ưu

### 3.4. Basic Feasible Solution (BFS)
- Một điểm trong feasible region với tối đa m biến (trong số n biến) khác 0
- Các biến khác 0 gọi là **basic variables**
- Các biến bằng 0 gọi là **non-basic variables**
- BFS tương ứng một-một với extreme points

### 3.5. Các Trường Hợp Đặc Biệt

**1. Infeasible (Không Khả Thi):**
- Không tồn tại điểm nào thỏa mãn tất cả ràng buộc
- Feasible region rỗng

**2. Unbounded (Không Bị Chặn):**
- Hàm mục tiêu có thể tiến đến -∞ (minimize) hoặc +∞ (maximize)
- Feasible region không bị chặn theo hướng của hàm mục tiêu

**3. Degenerate (Suy Biến):**
- Một BFS có nhiều hơn (n-m) biến bằng 0
- Có thể gây cycling trong Simplex algorithm

## 4. Phương Pháp Giải

### 4.1. Simplex Method (Phương Pháp Đơn Hình)

**Nguyên lý:**
- Bắt đầu từ một BFS (extreme point)
- Di chuyển dọc theo các cạnh của feasible region đến extreme point kề
- Chỉ di chuyển nếu giá trị hàm mục tiêu được cải thiện
- Dừng khi không thể cải thiện thêm (optimal) hoặc phát hiện unbounded

**Thuật toán (tóm tắt):**
```
1. Tìm một BFS khởi đầu (thường dùng Phase I)
2. Kiểm tra tính tối ưu:
   - Nếu tất cả reduced costs ≥ 0 (minimize) → OPTIMAL
   - Ngược lại, chọn biến non-basic để đưa vào basis (entering variable)
3. Tính direction of movement và step size
4. Xác định biến rời khỏi basis (leaving variable)
5. Cập nhật basis và BFS mới
6. Quay lại bước 2
```

**Độ phức tạp:**
- Worst-case: Exponential time O(2ⁿ)
- Average-case: Rất nhanh trong thực tế (polynomial behavior)
- Revised Simplex: Cải tiến với matrix operations hiệu quả hơn

### 4.2. Interior Point Method (Phương Pháp Điểm Trong)

**Nguyên lý:**
- Di chuyển qua **bên trong** feasible region (không đi dọc biên)
- Sử dụng barrier functions để tránh vi phạm ràng buộc
- Tiếp cận nghiệm tối ưu từ bên trong

**Ưu điểm:**
- Độ phức tạp polynomial: O(n³L) với L là số bit input
- Hiệu quả với bài toán large-scale (nhiều biến và ràng buộc)
- Convergence rate tốt (số vòng lặp ít phụ thuộc vào kích thước bài toán)

**Thuật toán phổ biến:**
- **Karmarkar's algorithm** (1984)
- **Primal-Dual Interior Point Method**
- **Barrier Method** với log barrier function

### 4.3. So Sánh Simplex vs Interior Point

| Aspect | Simplex | Interior Point |
|--------|---------|----------------|
| **Complexity** | Exponential (worst-case) | Polynomial |
| **Practical Speed** | Rất nhanh (small-medium) | Tốt hơn (large-scale) |
| **Warm Start** | ✓ Hỗ trợ tốt | ✗ Khó warm start |
| **Sensitivity Analysis** | ✓ Dễ dàng | ✗ Khó khăn |
| **Path** | Đi theo biên | Đi qua bên trong |
| **Số vòng lặp** | Phụ thuộc n, m | Ít phụ thuộc kích thước |

### 4.4. Dual Problem (Bài Toán Đối Ngẫu)

Mỗi bài toán LP (primal) có một bài toán đối ngẫu (dual):

**Primal:**
```
Minimize:    c'x
Subject to:  Ax ≥ b
             x ≥ 0
```

**Dual:**
```
Maximize:    b'y
Subject to:  A'y ≤ c
             y ≥ 0
```

**Các định lý quan trọng:**

1. **Weak Duality**: c'x ≥ b'y cho mọi feasible x, y
2. **Strong Duality**: Nếu primal có optimal x*, dual có optimal y* với c'x* = b'y*
3. **Complementary Slackness**: 
   - xⱼ > 0 ⟹ (A'y*)ⱼ = cⱼ
   - (A'y*)ⱼ < cⱼ ⟹ x*ⱼ = 0

**Ứng dụng dual:**
- Giải thích kinh tế (shadow prices)
- Kiểm tra tính tối ưu
- Một số bài toán dễ giải hơn ở dạng dual

## 5. Ưu Điểm của Linear Programming

### 5.1. Lý Thuyết Vững Chắc
- **Đảm bảo tối ưu toàn cục**: Nếu tồn tại nghiệm, LP tìm được global optimum
- **Cơ sở toán học chặt chẽ**: Dựa trên lý thuyết convex optimization
- **Duality theory**: Cung cấp lower/upper bounds và sensitivity analysis

### 5.2. Hiệu Quả Tính Toán
- **Thuật toán trưởng thành**: Simplex và Interior Point được tối ưu qua nhiều thập kỷ
- **Thời gian giải nhanh**: Polynomial time (Interior Point) hoặc rất nhanh trong thực tế (Simplex)
- **Scalability**: Giải được bài toán với hàng triệu biến và ràng buộc
- **Solver sẵn có**: CPLEX, Gurobi, HiGHS, GLPK miễn phí và thương mại

### 5.3. Tính Diễn Giải Cao (Interpretability)
- **Transparent**: Dễ hiểu logic của nghiệm (biến nào active, constraint nào binding)
- **Sensitivity analysis**: Phân tích ảnh hưởng của thay đổi parameters
- **Shadow prices**: Dual variables cho biết giá trị biên của resources
- **What-if scenarios**: Dễ dàng test các kịch bản khác nhau

### 5.4. Tính Ổn Định
- **Numerical stability**: Các solver hiện đại xử lý tốt vấn đề số học
- **Convergence đảm bảo**: Không bị stuck ở local optima như non-convex optimization
- **Robust**: Ít nhạy cảm với khởi tạo (trừ Simplex cần Phase I)

### 5.5. Tính Mở Rộng
- **Nền tảng cho MIP**: Mixed Integer Programming mở rộng LP với biến nguyên
- **Multi-objective**: Có thể kết hợp nhiều objectives (weighted sum, goal programming)
- **Stochastic programming**: Mở rộng với uncertainty
- **Network flow problems**: Dạng đặc biệt của LP giải rất hiệu quả

## 6. Hạn Chế của Linear Programming

### 6.1. Giả Định Tuyến Tính (Linearity Assumption)
- **Objective phải tuyến tính**: Không mô hình hóa được economies of scale, diminishing returns
- **Constraints phải tuyến tính**: Không biểu diễn được quan hệ phi tuyến (quadratic, exponential)
- **Fixed coefficients**: Không thích nghi với thay đổi môi trường
- **Superposition**: Giả định tổng hợp tuyến tính không luôn hợp lý trong thực tế

### 6.2. Giả Định Chia Nhỏ Được (Divisibility)
- **Fractional solutions**: LP cho phép nghiệm thực (0.5 xe, 3.7 công nhân)
- **Integer requirement**: Nhiều bài toán thực tế cần nghiệm nguyên
- **Rounding heuristics**: Làm tròn LP solution có thể không optimal hoặc infeasible
- **ILP complexity**: Integer Linear Programming là NP-hard (exponential time)

### 6.3. Giả Định Chắc Chắn (Certainty)
- **Deterministic**: Giả định parameters (c, A, b) biết chính xác
- **No uncertainty**: Không xử lý được random fluctuations
- **Static**: Không thay đổi theo thời gian
- **Robust optimization cần**: Phải mở rộng sang stochastic/robust LP

### 6.4. Tính Độc Lập (Independence)
- **No interaction**: Giả định các biến không tương tác phi tuyến
- **Separability**: Objective là tổng các hàm đơn biến
- **No correlation**: Không model được dependencies giữa resources

### 6.5. Hạn Chế Về Modeling

**1. Không Mô Hình Được:**
- Logical constraints (if-then-else) → cần MIP với binary variables
- Disjunctive constraints (A OR B) → cần MIP
- Non-convex regions → cần global optimization
- Dynamic systems → cần dynamic programming hoặc optimal control

**2. Myopic Optimization:**
- **Single-period**: LP giải từng bước độc lập
- **No foresight**: Không nhìn xa về tương lai (unlike DP, RL)
- **No learning**: Không thích nghi với historical patterns

**3. Trade-offs:**
- **Single objective**: Khó tối ưu nhiều mục tiêu mâu thuẫn đồng thời
- **Fixed weights**: Multi-objective LP cần xác định trọng số trước
- **Pareto frontier**: Không tự động tìm được tất cả Pareto optimal solutions

### 6.6. Hạn Chế Tính Toán

**1. Large-Scale Problems:**
- Mặc dù scalable, bài toán cực lớn (hàng tỷ biến) vẫn khó
- Memory requirements với ma trận dày đặc
- Numerical issues với ill-conditioned matrices

**2. Warm Start:**
- Interior Point khó warm start (không như Simplex)
- Re-optimization từ đầu khi thay đổi nhỏ

**3. Sensitivity:**
- Nghiệm có thể nhảy đột ngột khi thay đổi parameters
- Degeneracy gây khó khăn cho sensitivity analysis

## 7. Extensions và Variants

### 7.1. Integer Linear Programming (ILP)
```
Minimize:    c'x
Subject to:  Ax ≥ b
             x ≥ 0
             xⱼ ∈ ℤ  (integer) hoặc xⱼ ∈ {0,1} (binary)
```
- **Complexity**: NP-hard
- **Methods**: Branch-and-Bound, Cutting Planes, Branch-and-Cut

### 7.2. Mixed Integer Programming (MIP)
- Một số biến integer, một số biến continuous
- Mô hình hóa logical constraints, fixed charges, disjunctions

### 7.3. Multi-objective Linear Programming
```
Minimize:    [c₁'x, c₂'x, ..., cₖ'x]
Subject to:  Ax ≥ b, x ≥ 0
```
- **Methods**: Weighted sum, ε-constraint, goal programming
- **Output**: Pareto frontier

### 7.4. Stochastic Programming
```
Minimize:    𝔼[c(ω)'x(ω)]
Subject to:  A(ω)x(ω) ≥ b(ω)  for all scenarios ω
```
- Xử lý uncertainty với probability distributions
- **Approaches**: Two-stage, multi-stage, chance constraints

### 7.5. Robust Optimization
```
Minimize:    max_{c ∈ U} c'x
Subject to:  Ax ≥ b  for all A ∈ U_A, b ∈ U_b
```
- Tối ưu cho worst-case scenario
- Không cần probability distributions

## 8. Công Cụ và Thư Viện

### 8.1. Commercial Solvers
- **CPLEX** (IBM): Industry standard, rất nhanh
- **Gurobi**: Hiện đại, performance cao
- **XPRESS** (FICO): Tối ưu cho large-scale

### 8.2. Open-Source Solvers
- **HiGHS**: Modern, high-performance, MIT license
- **GLPK** (GNU Linear Programming Kit): Miễn phí, widely used
- **CLP** (COIN-OR): Part of COIN-OR suite
- **SoPlex**: Academic, numerical precision cao

### 8.3. Modeling Languages & Interfaces
- **Python**: scipy.optimize.linprog, PuLP, Pyomo, CVXPY
- **MATLAB**: linprog, optimization toolbox
- **Julia**: JuMP (Julia for Mathematical Programming)
- **AMPL, GAMS**: Specialized modeling languages

## 9. Kết Luận

### 9.1. Khi Nào Sử Dụng LP?
**LP phù hợp khi:**
- Bài toán có thể mô hình hóa tuyến tính
- Cần nghiệm tối ưu toàn cục với đảm bảo
- Thời gian giải nhanh là quan trọng
- Cần sensitivity analysis và interpretability
- Fractional solutions chấp nhận được

### 9.2. Khi Nào Cần Alternatives?
**Cần phương pháp khác khi:**
- Quan hệ phi tuyến (→ Nonlinear Programming)
- Biến nguyên bắt buộc (→ Integer Programming)
- Uncertainty quan trọng (→ Stochastic Programming)
- Dynamic, sequential decisions (→ Dynamic Programming, Reinforcement Learning)
- Objectives non-convex (→ Global Optimization, Metaheuristics)
- Learning from data (→ Machine Learning, Reinforcement Learning)

### 9.3. Vai Trò trong Optimization Landscape
Linear Programming là:
- **Nền tảng** của Operations Research
- **Baseline benchmark** cho các phương pháp phức tạp hơn
- **Building block** của nhiều algorithms (Branch-and-Bound, Cutting Planes)
- **Giải pháp thực tiễn** cho hàng ngàn ứng dụng công nghiệp mỗi ngày

## 10. Tài Liệu Tham Khảo

### 10.1. Sách Giáo Khoa Cơ Bản
1. **Dantzig, G. B. (1963).** *Linear Programming and Extensions*. Princeton University Press.
   - Tài liệu gốc từ người phát minh Simplex algorithm

2. **Bertsimas, D., & Tsitsiklis, J. N. (1997).** *Introduction to Linear Optimization*. Athena Scientific.
   - Giáo trình toàn diện, dễ hiểu cho người mới học

3. **Vanderbei, R. J. (2014).** *Linear Programming: Foundations and Extensions* (4th ed.). Springer.
   - Cân bằng giữa lý thuyết và thực hành

### 10.2. Sách Nâng Cao
4. **Luenberger, D. G., & Ye, Y. (2008).** *Linear and Nonlinear Programming* (3rd ed.). Springer.
   - Phương pháp Interior Point và các thuật toán hiện đại

5. **Boyd, S., & Vandenberghe, L. (2004).** *Convex Optimization*. Cambridge University Press.
   - Available free: https://web.stanford.edu/~boyd/cvxbook/
   - LP trong context của convex optimization

6. **Nocedal, J., & Wright, S. J. (2006).** *Numerical Optimization* (2nd ed.). Springer.
   - Chi tiết về numerical methods và implementation

### 10.3. Tài Liệu Kỹ Thuật
7. **scipy.optimize.linprog documentation**
   - https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html

8. **HiGHS solver**
   - https://github.com/ERGO-Code/HiGHS
   - High-performance open-source LP/MIP solver

9. **COIN-OR project**
   - https://www.coin-or.org/
   - Open-source optimization software suite

### 10.4. Courses & Tutorials
10. **MIT OpenCourseWare: 15.053 Optimization Methods**
    - https://ocw.mit.edu/courses/15-053-optimization-methods-in-management-science-spring-2013/

11. **Stanford: Convex Optimization (EE364a)**
    - https://web.stanford.edu/class/ee364a/
