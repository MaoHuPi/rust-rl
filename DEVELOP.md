# DEVELOP

## Develop Notes

* 20240101 - A000001

  Update about Gradient Descent  
  階段性錯誤說明與重新推導  
  因為 self.value 在 self.calc_value 函式中所做的是 $\sigma\left(\sum{{v_i + w_i} + b}\right)$ ，  
  所以 cost 為 $\left(self.value - anticipated\_ value\right)^2$ ，  
  即是 $\left(\sigma(\sum{\left(v_i + w_i\right)} + b) - anticipated\_ value\right)^2$ 。

  因此如果是以 $\sigma$ 的反函數來進行初步處理的話，應會變成 $\left(\sigma^{-1}\left(\sigma\left(\sum{\left(v_i + w_i\right)} + b\right)\right) - \sigma^{-1} \times \left(anticipated\_ value\right)\right)^2$，  
  而非 $\left(\sigma\left(\sum{\left(v_i + w_i\right)} + b\right) - \sigma^{-1}\left(anticipated\_ value\right)\right)^2$ 。

  至於最正確的做法應該是直接對 cost 本身求導，  
  而不是在 cost 內部又進行啟動函數反函數的先行處理，  
  所以應該這樣堆導：  
  $$
  \begin{equation}
  \begin{align*}
  \frac{\partial}{\partial w_i}cost
  & = \frac{\partial}{\partial w_i}\left(\sigma\left(\sum{\left(v_i + w_i\right)} + b\right) - anticipated\_ value\right)^2\\
  & \left(let\ A(v_i, w_i, b) = \sum{\left(v_i + w_i\right)} + b,\ B(u) = \sigma(u),\ C(v) = \left(v - anticipated\_ value\right)^2\right)\\
  & = \frac{dC}{dv} \times \frac{dB}{du} \times \frac{\partial A}{\partial w_i}\\
  & = (2\times self.value-2\times anticipated\_ value) \times (self.value\times (1-self.value)) \times (v_i)
  \end{align*}
  \end{equation}
  $$

* 20240102 - A000002

  Question about Gradient Descent  
  輸出層除外的其他節點層如果也直接使用目前的方法計算的話應該會有很大的問題，  
  目前多是以「層數少」、「線性問題」、「單輸出」的方式進行測試，  
  而如果有兩個以上的輸出，或者隱藏層層數增加，那是不是會在單節點單權重的梯度運算上出現算法上的差異？  
  待查證與計算。  
  （也難怪只要有兩個輸出就會出現不管輸入是什麼 $o_1$ 總是接近一個數字 $o_2$ 則接近另一個數字的結果？）

* 20240105 - A000003

  Solution of A000002  
  在以下結構的類神經網路中

  ```mermaid
  graph LR;
    classDef node fill:black,stroke:white,stroke-width:1px

    i1(("i1")):::node;i2(("i2")):::node
    v1(("v1")):::node;v2(("v2")):::node;v3(("v3")):::node;v4(("v4")):::node
    o(("o")):::node

    i1-->|w1|v1;i2-->|w2|v1
    i1-->|w3|v2;i2-->|w4|v2
    v1==>|w5|v3;v2-->|w6|v3
    v1==>|w7|v4;v2-->|w8|v4
    v3==>|w9|o;v4==>|w10|o
  ```

  $\frac{\partial v_3}{\partial o}$ 、 $\frac{\partial v_4}{\partial o}$ 、 $\frac{\partial v_1}{\partial v_3}$、 $\frac{\partial v_1}{\partial v_4}$ 皆可以使用目前的「以一神經元之參數對該神經元輸出做偏微分」的方式求得。  
  而根據chain rule可知 $\frac{\partial v_1}{\partial o}=\frac{\partial v_1}{\partial v_3}\times \frac{\partial v_3}{\partial o} + \frac{\partial v_1}{\partial v_4}\times \frac{\partial v_4}{\partial o}$ ，  
  使用這樣子直接以要調整之參數對 lost function 做偏微分的方式，  
  可以更精確地對參數做調整，而非只是進行個別的 cost minimal。