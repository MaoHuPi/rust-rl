# DEVELOP

## Develop Notes

* 20240101 - A000001

```txt
Update about Gradient Descent
階段性錯誤說明與重新推導
因為 self.value 在 self.calc_value 函式中所做的是 F_a(sum{v_i + w_i} + b) ，
所以 cost 為 pow{self.value - anticipated_value, 2} ，
即是 pow{F_a(sum{v_i + w_i} + b) - anticipated_value, 2} 。

因此如果是以 F_a 的反函數來進行初步處理的話，應會變成 pow{f_a_inv(F_a(sum{v_i + w_i} + b)) - f_a_inv(anticipated_value), 2} ，
而非 pow{F_a(sum{v_i + w_i} + b) - f_a_inv(anticipated_value), 2} 。

至於最正確的做法應該是直接對 cost 本身求導，
而不是在 cost 內部又進行啟動函數反函數的先行處理，
所以應該這樣堆導：
partial{cost, w_i}
 = partial{pow{F_a(sum{v_i + w_i} + b) - anticipated_value, 2}, w_i}
   (let A(v_i, w_i, b) = sum{v_i + w_i} + b, B(u) = F_a(u), C(v) = pow{v - anticipated_value, 2})
 = der{C, v} * der{B, u} * partial{A, w_i}
 = (2*self.value-2anticipated_value) * (self.value*(1-self.value)) * (v_i)
```

* 20240102 - A000002

```txt
Question about Gradient Descent
輸出層除外的其他節點層如果也直接使用目前的方法計算的話應該會有很大的問題，
目前多是以「層數少」、「線性問題」、「單輸出」的方式進行測試，
而如果有兩個以上的輸出，或者隱藏層層數增加，那是不是會在單節點單權重的梯度運算上出現算法上的差異？
待查證與計算。
（也難怪只要有兩個輸出就會出現不管輸入是什麼 o_1 總是接近一個數字 o_2 則接近另一個數字的結果？）
```