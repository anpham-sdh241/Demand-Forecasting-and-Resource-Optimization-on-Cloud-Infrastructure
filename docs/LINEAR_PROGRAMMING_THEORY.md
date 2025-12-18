# Lý Thuyết Linear Programming (Quy Hoạch Tuyến Tính)

## 1. Giới Thiệu

Linear Programming (LP) hay Quy Hoạch Tuyến Tính là một kỹ thuật trong đại số sử dụng các phương trình tuyến tính để xác định cách đạt được tình huống tối ưu (tối đa hoặc tối thiểu) như một câu trả lời cho bài toán toán học, giả định tính hữu hạn của tài nguyên và tính định lượng của mục tiêu tối ưu hóa cuối cùng. [[1]](https://www.spiceworks.com/soft-tech/linear-programming/) [[2]](https://www.geeksforgeeks.org/maths/linear-programming/)

LP sử dụng nhiều bất đẳng thức tuyến tính liên quan đến một tình huống nhất định để xác định giá trị "tối ưu" có thể đạt được dưới các ràng buộc đó. Một ví dụ điển hình là tính toán mức sản xuất "tối ưu" để tối đa hóa lợi nhuận, với các hạn chế về nguồn cung và nhân sự.

**Tại sao Linear Programming cần thiết?**

Trong thực tế, chúng ta thường gặp các tình huống dựa trên mục tiêu hàng ngày với tài nguyên hạn chế. LP giúp giải quyết các vấn đề tối ưu hóa bằng cách:
- Tối đa hóa lợi ích (profit, efficiency)
- Tối thiểu hóa chi phí (cost, time, resource usage)
- Phân bổ tài nguyên một cách tối ưu

LP là nền tảng cho các mô hình toán học đại diện cho các kết nối thực tế, đặc biệt quan trọng trong **phân bổ tài nguyên (resource allocation)** và **lập lịch (scheduling)**. [[1]](https://www.spiceworks.com/soft-tech/linear-programming/)

## 2. Dạng Toán Học Tổng Quát

### 2.1. Các Thành Phần Cơ Bản

Một bài toán Linear Programming bao gồm 4 thành phần chính: [[1]](https://www.spiceworks.com/soft-tech/linear-programming/) [[2]](https://www.geeksforgeeks.org/maths/linear-programming/)

1. **Decision Variables (Biến quyết định)**: Các biến cần xác định để đạt được nghiệm tối ưu. Trong bài toán phân bổ tài nguyên, đây thường là số lượng tài nguyên được phân bổ cho từng task/resource type.
   - Ví dụ: xᵢ = số lượng VM loại i được phân bổ

2. **Objective Function (Hàm mục tiêu)**: Phương trình đại số biểu diễn mục tiêu cần đạt được (tối đa hoặc tối thiểu)
   - Ví dụ: Minimize Z = Σ(cᵢ × xᵢ) - tổng chi phí phân bổ tài nguyên

3. **Constraints (Ràng buộc)**: Các giới hạn hoặc hạn chế mà biến quyết định phải tuân theo
   - Ví dụ: Σ(CPUᵢ × xᵢ) ≥ CPU_required - ràng buộc về nhu cầu CPU

4. **Non-Negativity Restrictions (Điều kiện không âm)**: Biến quyết định phải có giá trị không âm
   - Ví dụ: xᵢ ≥ 0 (số lượng tài nguyên không thể âm)

### 2.2. Dạng Toán Học Đơn Giản

Với bài toán có 2 biến quyết định, dạng toán học có thể viết như sau:

**Hàm mục tiêu:**
```
Z = ax + by
```

**Ràng buộc:**
```
cx + dy ≤ e    (hoặc ≥, =)
px + qy ≤ r    (hoặc ≥, =)
...
```

**Điều kiện không âm:**
```
x ≥ 0, y ≥ 0
```

### 2.3. Dạng Tổng Quát (n biến)

Với n biến quyết định, bài toán LP có dạng:

**Minimize (hoặc Maximize):**
```
Z = c₁x₁ + c₂x₂ + ... + cₙxₙ
```

**Subject to (Ràng buộc):**
```
a₁₁x₁ + a₁₂x₂ + ... + a₁ₙxₙ  {≤, =, ≥}  b₁
a₂₁x₁ + a₂₂x₂ + ... + a₂ₙxₙ  {≤, =, ≥}  b₂
...
aₘ₁x₁ + aₘ₂x₂ + ... + aₘₙxₙ  {≤, =, ≥}  bₘ
```

**Điều kiện không âm:**
```
x₁, x₂, ..., xₙ ≥ 0
```

**Dạng ma trận:**
```
Minimize:    Z = c'x
Subject to:  Ax {≤, =, ≥} b
             x ≥ 0
```

Trong đó:
- **x** = [x₁, x₂, ..., xₙ]ᵀ: Vector biến quyết định
- **c** = [c₁, c₂, ..., cₙ]ᵀ: Vector hệ số mục tiêu
- **A**: Ma trận hệ số ràng buộc (m×n)
- **b**: Vector vế phải (m×1)

### 2.4. Đặc Điểm của Bài Toán Linear Programming

Một bài toán được giải bằng LP phải có các đặc điểm sau: [[1]](https://www.spiceworks.com/soft-tech/linear-programming/)

1. **Subject to constraints (Chịu ràng buộc)**: Các hạn chế về tài nguyên phải được biểu diễn dưới dạng toán học

2. **Geared towards an objective function (Hướng tới hàm mục tiêu)**: Hàm mục tiêu của bài toán phải được mô tả định lượng

3. **A linear relationship (Quan hệ tuyến tính)**: Mối quan hệ giữa các biến độc lập phải là tuyến tính (bậc một)

4. **Includes only finite numbers (Chỉ bao gồm số hữu hạn)**: Phải có số lượng input và output hữu hạn

5. **Does not include negative values (Không bao gồm giá trị âm)**: Giá trị biến phải bằng 0 hoặc dương

6. **Hinges on decision variables (Phụ thuộc vào biến quyết định)**: Kết quả được xác định bởi biến quyết định, cung cấp nghiệm cuối cùng cho bài toán

### 2.5. Khái Niệm Cơ Bản

- **Feasible Region (Miền chấp nhận được)**: Tập hợp tất cả các điểm thỏa mãn tất cả ràng buộc. Đây là một tập lồi (convex set), thường được gọi là "feasibility region". [[1]](https://www.spiceworks.com/soft-tech/linear-programming/)

- **Optimal Solution (Nghiệm tối ưu)**: Điểm trong feasible region có giá trị hàm mục tiêu tốt nhất (tối đa hoặc tối thiểu). Nghiệm tối ưu thường nằm tại các điểm cực trị (extreme points) của feasible region.

- **Viable Solution (Nghiệm khả thi)**: Tập hợp tất cả các nghiệm tiềm năng thỏa mãn các ràng buộc dưới dạng biến.

- **Optimum Value (Giá trị tối ưu)**: Nghiệm tốt nhất có thể hỗ trợ hiệu quả mục tiêu của bài toán.

- **Các trường hợp đặc biệt:**
  - **Infeasible (Không khả thi)**: Không tồn tại điểm nào thỏa mãn tất cả ràng buộc
  - **Unbounded (Không bị chặn)**: Hàm mục tiêu có thể tiến đến -∞ (minimize) hoặc +∞ (maximize)

## 3. Linear Programming cho Bài Toán Allocation/Rescheduling

### 3.1. Ứng Dụng trong Resource Allocation

Linear Programming đặc biệt hiệu quả cho các bài toán **phân bổ tài nguyên (resource allocation)** và **lập lịch (scheduling/rescheduling)**. [[1]](https://www.spiceworks.com/soft-tech/linear-programming/)

**Ví dụ bài toán phân bổ tài nguyên:**

Giả sử cần phân bổ các loại VM (Virtual Machine) để đáp ứng nhu cầu workload:

- **Biến quyết định**: xᵢ = số lượng VM loại i được phân bổ (i = 1, 2, ..., n)
- **Hàm mục tiêu**: Minimize Z = Σ(cᵢ × xᵢ) - tổng chi phí vận hành VM
- **Ràng buộc**:
  - Ràng buộc CPU: Σ(CPUᵢ × xᵢ) ≥ CPU_required
  - Ràng buộc Memory: Σ(Memoryᵢ × xᵢ) ≥ Memory_required
  - Ràng buộc capacity: xᵢ ≤ Max_availableᵢ (nếu có giới hạn)
- **Điều kiện không âm**: xᵢ ≥ 0

### 3.2. Đặc Điểm của Bài Toán Allocation

Bài toán phân bổ tài nguyên thường có các đặc điểm:

1. **Multiple resource types**: Nhiều loại tài nguyên (CPU, Memory, Storage, etc.)
2. **Capacity constraints**: Ràng buộc về khả năng của từng loại tài nguyên
3. **Demand requirements**: Nhu cầu tối thiểu phải được đáp ứng
4. **Cost optimization**: Tối thiểu hóa chi phí hoặc tối đa hóa hiệu quả

### 3.3. Rescheduling với Linear Programming

Trong bài toán **rescheduling** (lập lịch lại), LP có thể được sử dụng để:

- Tối ưu hóa phân bổ tài nguyên theo thời gian
- Điều chỉnh lịch trình khi có thay đổi nhu cầu
- Tối thiểu hóa chi phí chuyển đổi (switching cost) giữa các cấu hình

**Mô hình rescheduling:**
- **Biến quyết định**: xᵢₜ = số lượng tài nguyên loại i tại thời điểm t
- **Hàm mục tiêu**: Minimize (operational cost + switching cost)
- **Ràng buộc**: 
  - Đáp ứng nhu cầu tại mỗi thời điểm
  - Giới hạn về khả năng thay đổi giữa các thời điểm

## 4. Phương Pháp Giải

### 4.1. Các Bước Giải Bài Toán LP

Các bước quan trọng nhất trong việc giải bài toán LP là xây dựng bài toán từ dữ liệu đã cho: [[1]](https://www.spiceworks.com/soft-tech/linear-programming/) [[2]](https://www.geeksforgeeks.org/maths/linear-programming/)

**Bước 1**: Xác định biến quyết định trong bài toán  
Ví dụ: xᵢ = số lượng VM loại i

**Bước 2**: Xây dựng hàm mục tiêu và xác định cần tối đa hay tối thiểu  
Ví dụ: Minimize Z = Σ(cᵢ × xᵢ)

**Bước 3**: Viết tất cả các ràng buộc của bài toán  
Ví dụ: Σ(CPUᵢ × xᵢ) ≥ CPU_required

**Bước 4**: Đảm bảo điều kiện không âm cho biến quyết định  
xᵢ ≥ 0

**Bước 5**: Giải bài toán bằng phương pháp Simplex, Graphical Method, hoặc sử dụng solver

### 4.2. Simplex Method (Phương Pháp Đơn Hình)

Simplex Method là phương pháp điển hình để giải các bài toán tối ưu hóa trong LP. [[1]](https://www.spiceworks.com/soft-tech/linear-programming/) [[2]](https://www.geeksforgeeks.org/maths/linear-programming/) [[3]](https://masarat-sy.org/en/simplex-method/)

**Nguyên lý:**
- Thường bao gồm một hàm và một số ràng buộc được viết dưới dạng bất đẳng thức
- Bất đẳng thức xác định một vùng đa giác, với nghiệm thường nằm tại một đỉnh
- Phương pháp này kiểm tra có hệ thống các đỉnh như các nghiệm tiềm năng
- Thông qua quy trình lặp, phương pháp tiếp cận và cuối cùng đạt được giá trị tối đa hoặc tối thiểu của hàm mục tiêu

**Các bước thực hiện:**

**Bước 1**: Xây dựng bài toán LP dựa trên các ràng buộc đã cho

**Bước 2**: Chuyển đổi tất cả bất phương trình thành phương trình bằng cách thêm biến slack (slack variable)
- Nếu a₁x₁ + a₂x₂ ≤ b, thêm s ≥ 0: a₁x₁ + a₂x₂ + s = b
- Nếu a₁x₁ + a₂x₂ ≥ b, thêm s ≥ 0: a₁x₁ + a₂x₂ - s = b

**Bước 3**: Xây dựng bảng Simplex ban đầu
- Mỗi phương trình ràng buộc là một hàng
- Viết hàm mục tiêu ở hàng cuối cùng

**Bước 4**: Xác định pivot column (cột xoay)
- Tìm entry âm lớn nhất trong hàng cuối (hàm mục tiêu)

**Bước 5**: Xác định pivot row (hàng xoay)
- Chia các entry của cột ngoài cùng bên phải với các entry tương ứng của pivot column
- Hàng có tỷ lệ dương nhỏ nhất là pivot row

**Bước 6**: Thực hiện phép biến đổi hàng (row operations) để cập nhật bảng Simplex

**Bước 7**: Lặp lại các bước 4-6 cho đến khi tất cả các entry trong hàng cuối đều không âm → đạt được nghiệm tối ưu

**Độ phức tạp:**
- Worst-case: Exponential O(2ⁿ)
- Average-case: Rất nhanh trong thực tế (polynomial behavior)

**Khả năng của Simplex Method:**
- Xác định các ràng buộc trùng lặp
- Tìm nghiệm hoàn chỉnh
- Xác định nhiều phương án thay thế
- Phát hiện bài toán không khả thi

### 4.3. Graphical Method (Phương Pháp Đồ Thị)

Phương pháp đồ thị chỉ áp dụng được khi bài toán có **2 biến quyết định**. [[1]](https://www.spiceworks.com/soft-tech/linear-programming/) [[2]](https://www.geeksforgeeks.org/maths/linear-programming/)

**Các bước:**
1. Vẽ các đường thẳng biểu diễn ràng buộc trên mặt phẳng tọa độ
2. Xác định feasible region (miền giao của tất cả ràng buộc)
3. Tìm các điểm cực trị (corner points) của feasible region
4. Tính giá trị hàm mục tiêu tại mỗi điểm cực trị
5. Điểm có giá trị tốt nhất (tối đa hoặc tối thiểu) là nghiệm tối ưu

**Phương pháp:**
- **Extreme or corner points method**: Kiểm tra giá trị hàm mục tiêu tại các điểm cực trị
- **Iso-profit (cost) efficiency line method**: Vẽ các đường song song với gradient của phương trình để xác định điểm kết hợp cho cùng chi phí/lợi nhuận

### 4.4. Interior Point Method (Phương Pháp Điểm Trong)

**Nguyên lý:**
- Di chuyển qua **bên trong** feasible region (không đi dọc biên như Simplex)
- Sử dụng barrier functions để tránh vi phạm ràng buộc
- Tiếp cận nghiệm tối ưu từ bên trong

**Ưu điểm:**
- Độ phức tạp polynomial: O(n³L) với L là số bit input
- Hiệu quả với bài toán large-scale (nhiều biến và ràng buộc)

**So sánh Simplex vs Interior Point:**

| Aspect | Simplex | Interior Point |
|--------|---------|----------------|
| **Complexity** | Exponential (worst-case) | Polynomial |
| **Practical Speed** | Rất nhanh (small-medium) | Tốt hơn (large-scale) |
| **Sensitivity Analysis** | ✓ Dễ dàng | ✗ Khó khăn |
| **Path** | Đi theo biên | Đi qua bên trong |

### 4.5. Sử Dụng Solver (Phương Pháp Phổ Biến Trong Thực Tế)

Trong thực tế, thay vì triển khai trực tiếp Simplex hoặc Interior Point, người ta thường sử dụng **solver** (phần mềm giải LP) đã được tối ưu hóa. Các solver hiện đại tự động chọn phương pháp phù hợp nhất cho từng bài toán.

**Các solver phổ biến:**

1. **HiGHS (High-performance solver)**: 
   - Open-source, high-performance solver
   - Được sử dụng trong `scipy.optimize.linprog` với `method="highs"`
   - Hỗ trợ cả Simplex và Interior Point methods
   - Tự động chọn phương pháp tối ưu dựa trên đặc điểm bài toán

2. **Gurobi, CPLEX**: Commercial solvers, rất nhanh và mạnh mẽ

3. **GLPK, CLP**: Open-source alternatives

**Ưu điểm của việc sử dụng solver:**
- **Tự động tối ưu**: Solver tự động chọn phương pháp tốt nhất (Simplex hoặc Interior Point)
- **Đã được tối ưu**: Các solver được tối ưu qua nhiều thập kỷ, hiệu quả cao
- **Dễ sử dụng**: Chỉ cần định nghĩa bài toán, không cần triển khai thuật toán
- **Xử lý edge cases**: Xử lý tốt các trường hợp đặc biệt (degenerate, infeasible, unbounded)

**Trong đề tài này:**
- Sử dụng `scipy.optimize.linprog` với `method="highs"` (HiGHS solver)
- HiGHS tự động chọn phương pháp phù hợp (có thể là Simplex hoặc Interior Point)
- Phù hợp cho bài toán phân bổ VM với nhiều biến và ràng buộc

## 5. Ưu Điểm và Hạn Chế

Linear Programming là một công cụ mạnh mẽ và được sử dụng rộng rãi trong nhiều lĩnh vực, đặc biệt là trong các bài toán phân bổ tài nguyên và lập lịch. Tuy nhiên, để áp dụng hiệu quả, cần hiểu rõ cả những ưu điểm và hạn chế của phương pháp này.

### 5.1. Ưu Điểm

Mặc dù có một số hạn chế, Linear Programming vẫn là một trong những phương pháp tối ưu hóa được sử dụng phổ biến nhất nhờ những ưu điểm nổi bật sau đây:

1. **Đảm bảo tối ưu toàn cục**: LP tìm được global optimum (không phải local optimum), đảm bảo nghiệm tốt nhất có thể. Đây là ưu điểm quan trọng so với các phương pháp heuristic hoặc metaheuristic chỉ tìm được nghiệm gần tối ưu.

2. **Hiệu quả tính toán**: 
   - Thuật toán trưởng thành, được tối ưu qua nhiều thập kỷ
   - Giải nhanh với solver hiện đại (HiGHS, Gurobi, CPLEX)
   - Có thể giải bài toán với hàng triệu biến và ràng buộc
   - Thời gian giải thường rất nhanh trong thực tế, mặc dù độ phức tạp worst-case có thể là exponential

3. **Tính diễn giải cao**: 
   - Dễ hiểu logic của nghiệm (biến nào active, ràng buộc nào binding)
   - Có sensitivity analysis để phân tích ảnh hưởng của thay đổi parameters
   - Shadow prices (dual variables) cho biết giá trị biên của resources, hỗ trợ ra quyết định

4. **Ứng dụng rộng rãi trong Allocation/Scheduling**: [[1]](https://www.spiceworks.com/soft-tech/linear-programming/)
   - Các công ty sản xuất sử dụng LP rộng rãi để tổ chức và lập lịch sản xuất
   - Dịch vụ giao hàng sử dụng LP để xác định tuyến đường ngắn nhất, giảm thời gian di chuyển và tiêu thụ nhiên liệu
   - Tổ chức tài chính thiết lập phạm vi công cụ đầu tư có thể được cung cấp cho khách hàng
   - Đặc biệt hiệu quả cho bài toán phân bổ tài nguyên cloud với nhiều loại VM và ràng buộc

5. **Cung cấp insights quan trọng**: LP cung cấp những hiểu biết quan trọng về các thách thức kinh doanh bằng cách tạo điều kiện xác định nghiệm tối ưu trong mỗi tình huống nhất định, giúp ra quyết định dựa trên dữ liệu.

### 5.2. Hạn Chế

Mặc dù có nhiều ưu điểm, Linear Programming cũng có những hạn chế nhất định do các giả định cơ bản của nó. Việc hiểu rõ các hạn chế này giúp xác định khi nào LP phù hợp và khi nào cần sử dụng các phương pháp thay thế:

1. **Giả định tuyến tính**: 
   - Hàm mục tiêu và ràng buộc phải tuyến tính (bậc một)
   - Không mô hình hóa được quan hệ phi tuyến (quadratic, exponential, economies of scale)
   - Chỉ áp dụng cho các bài toán có hàm mục tiêu và ràng buộc tuyến tính
   - Trong thực tế, nhiều quan hệ không phải tuyến tính (ví dụ: chi phí giảm dần khi mua số lượng lớn)

2. **Giả định chia nhỏ được (Divisibility)**:
   - LP cho phép nghiệm thực (ví dụ: 2.5 đơn vị, 3.7 công nhân)
   - Nhiều bài toán thực tế cần nghiệm nguyên (số lượng VM phải là số nguyên) → cần Integer Linear Programming (ILP), là NP-hard và khó giải hơn nhiều
   - Làm tròn nghiệm có thể không optimal hoặc thậm chí infeasible
   - Trong bài toán phân bổ VM, cần làm tròn nghiệm từ LP, có thể dẫn đến lãng phí tài nguyên

3. **Giả định chắc chắn (Certainty)**:
   - Parameters (c, A, b) phải biết chính xác
   - Không xử lý được uncertainty trong nhu cầu hoặc chi phí
   - Trong thực tế, nhu cầu tài nguyên thường không chắc chắn → cần Stochastic Programming hoặc Robust Optimization
   - Bài toán phân bổ VM với dự báo không chính xác có thể dẫn đến phân bổ không tối ưu

4. **Hạn chế về modeling**:
   - Single-period: Giải từng bước độc lập, không nhìn xa về tương lai
   - Không tối ưu chi phí chuyển đổi (switching cost) giữa các bước thời gian một cách toàn cục
   - Single objective: Khó tối ưu nhiều mục tiêu mâu thuẫn đồng thời (cần Multi-objective LP)
   - Khó mô hình hóa các ràng buộc logic phức tạp (if-then-else) mà không cần binary variables

5. **Độ phức tạp trong trường hợp xấu**: 
   - Simplex method có thể yêu cầu số lượng bước lặp lớn trong một số trường hợp đặc biệt (degenerate cases), dẫn đến thời gian tính toán dài
   - Mặc dù trong thực tế thường rất nhanh, worst-case complexity là exponential

## Tài Liệu Tham Khảo

1. **Spiceworks - Linear Programming**  
   https://www.spiceworks.com/soft-tech/linear-programming/  
   *(Định nghĩa, đặc điểm, phương pháp giải và ứng dụng trong resource allocation)*

2. **GeeksforGeeks - Linear Programming**  
   https://www.geeksforgeeks.org/maths/linear-programming/  
   *(Giới thiệu tổng quan, các thành phần, công thức và phương pháp giải)*

3. **Masarat - Simplex Method**  
   https://masarat-sy.org/en/simplex-method/  
   *(Lịch sử phát triển và ứng dụng thực tế của phương pháp Simplex)*

4. **scipy.optimize.linprog documentation**  
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html  
   *(Hướng dẫn sử dụng solver LP trong Python)*

5. **Vanderbei, R. J. (2014).** *Linear Programming: Foundations and Extensions* (4th ed.). Springer.  
   *(Giáo trình toàn diện về LP)*
