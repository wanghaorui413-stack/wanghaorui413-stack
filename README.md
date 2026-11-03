<div align="center">

<img src="assets/header.svg" width="100%" alt="Austin Wang · AI 产品经理 · 可控 AI 工具 · 数据驱动" />

<a href="mailto:13701813467@163.com"><img src="https://img.shields.io/badge/邮箱-13701813467%40163.com-0EA5E9?style=flat-square&logo=maildotru&logoColor=white" alt="邮箱" /></a>
<img src="https://img.shields.io/badge/求职方向-AI%20产品经理-22C55E?style=flat-square" alt="求职方向：AI 产品经理" />
<a href="https://github.com/wanghaorui413-stack?tab=repositories"><img src="https://img.shields.io/badge/全部项目-→-1E293B?style=flat-square&logo=github" alt="全部项目" /></a>

</div>

## 👋 你好，我是 Austin

我在华东师范大学读物理海洋学硕士，2027 届，正在准备秋招。

平时学习和做科研，总会遇到一些"要是有个工具就好了"的时刻。于是我开始用 vibecoding 给自己做工具：雅思写作练习没有趁手的地方，就做了一个写作台；备考和科研总是顾此失彼，并且还容易错过测评、AI面、真人面等信息，因此我搭建了一个人工作台用于分配我的时间处理多项任务并提醒我待办事宜。

除此以外我还搭建了一些工具用于提高生活、工作和学习的效率。

另外我做过海洋数据方面的研究，包括地形 Rossby 波，最近在做叶绿素浓度的遥感反演。

## 🚀 做过的项目

### 🧳 [AI Trip Planner](https://github.com/wanghaorui413-stack/ai-trip-planner) · RAG 旅行规划

<table>
  <tr>
    <td width="50%"><img src="https://raw.githubusercontent.com/wanghaorui413-stack/ai-trip-planner/main/assets/screenshots/home.png" alt="填写目的地、预算和偏好" /></td>
    <td width="50%"><img src="https://raw.githubusercontent.com/wanghaorui413-stack/ai-trip-planner/main/assets/screenshots/results.png" alt="多套行程方案对比" /></td>
  </tr>
</table>

填好目的地、天数、预算和偏好，它会给出几套可以比较的行程，每套都带预算拆分和每天的路线。

通用大模型很容易给出"好玩但超预算"的方案，所以 RAG 在这里只负责补充推荐理由，预算核算、排序和校验都交给规则。我还写了 [PRD](https://github.com/wanghaorui413-stack/ai-trip-planner/blob/main/docs/PRD.md) 和一套[离线评测](https://github.com/wanghaorui413-stack/ai-trip-planner/blob/main/docs/EVALUATION.md)：24 个合成场景里通过了 20 个，平均分 0.943。没通过的 4 个都是"体验优先"方案超预算不到 5%，这是下一步要修的地方。

`Python` `FastAPI` `Streamlit` `FAISS` `LangChain` `Docker`

<sub>目前用的是 mock 数据，评测结果不代表真实用户表现。</sub>

### 🌊 [Chlorophyll Lab](projects/chlorophyll-lab) · 叶绿素 a 浓度研究

<p align="center"><img src="assets/chlorophyll-preview.png" width="88%" alt="Sentinel-2 校正反射率与有效像元浓度估计" /></p>

这是我的科研项目：用水体的光谱估算叶绿素 a 浓度。

我在 GLORIA 实测光谱（质控后 3,385 条）上比较了 MLP 和 1D CNN，保留集 log10 RMSE 是 0.2387。另外搭了一条卫星处理流程：Sentinel-2 原始影像经过 ACOLITE 大气校正，输出带质量掩膜和溯源信息的 20 m 浓度图。在汉江的样例里，严格质检后只剩 124 / 35,055 个像元可用，所以它证明的是流程能跑通，还不能当成一张完整的水质图。项目里的 32 项本地测试都能通过。

`Python` `PyTorch` `scikit-learn` `Rasterio` `Sentinel-2` `ACOLITE`

<sub>log10 RMSE 是历史开发评估，不是外部盲测；卫星浓度的精度还要和现场实测数据对比验证。</sub>

### 🖐️ [手势驱动的 3D 记忆相册](https://github.com/wanghaorui413-stack/gesture-memory-gallery) · Gesture Memory Gallery

<table>
  <tr>
    <td width="50%"><img src="https://raw.githubusercontent.com/wanghaorui413-stack/gesture-memory-gallery/main/docs/assets/preview-desktop.png" alt="粒子蛋糕主场景" /></td>
    <td width="50%"><img src="https://raw.githubusercontent.com/wanghaorui413-stack/gesture-memory-gallery/main/docs/assets/preview-scatter.jpg" alt="张开手，照片从蛋糕里散开" /></td>
  </tr>
</table>

一开始是给妹妹做的生日礼物。打开是一个由粒子组成的 3D 蛋糕：张开手，照片从蛋糕里散开；握拳收回；捏合手指放大其中一张。

手势识别用 MediaPipe 直接在浏览器里跑。为了减少手抖带来的误触，我按掌宽来定判断阈值，并且要求连续几帧都识别到同一个手势才算数。没有摄像头也能用按钮和触屏操作。做好之后可以打包成一个 HTML 文件，对方双击就能打开。

`Three.js` `MediaPipe` `Vite` `JavaScript` `Canvas`

<sub>截图里的照片是示意图。</sub>

### 🧩 两个顺手做的小工具

<table>
  <tr>
    <td width="50%" valign="top">
      <img src="https://raw.githubusercontent.com/wanghaorui413-stack/IELTS-Writing/main/assets/screenshots/01-dashboard.png" alt="IELTS Writing Coach 写作工作台" />
      <p><a href="https://github.com/wanghaorui413-stack/IELTS-Writing"><b>IELTS Writing Coach</b></a><br />备考雅思时做的写作工作台。选题、写前提纲、限时写作和自查放在同一个页面里，写的时候就能看到词数、连接词和常见语法错误的提醒。</p>
    </td>
    <td width="50%" valign="top">
      <img src="https://raw.githubusercontent.com/wanghaorui413-stack/self-work-station/main/docs/screenshots/01-overview.png" alt="Daily Balance Planner 每日计划" />
      <p><a href="https://github.com/wanghaorui413-stack/self-work-station"><b>Daily Balance Planner</b></a><br />同时备考和做科研，经常一整天只顾一头。它先给每个任务留出最低时间，剩下的按权重分配，再用近 7 天的记录找出总被跳过的那一项。</p>
    </td>
  </tr>
</table>

## 🧰 常用工具

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain" />
  <img src="https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white" alt="FAISS" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript" />
  <img src="https://img.shields.io/badge/Three.js-000000?style=for-the-badge&logo=threedotjs&logoColor=white" alt="Three.js" />
  <img src="https://img.shields.io/badge/MATLAB-0076A8?style=for-the-badge&logo=mathworks&logoColor=white" alt="MATLAB" />
</p>

## 🐍 贡献轨迹

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/wanghaorui413-stack/wanghaorui413-stack/gh-pages/github-contribution-grid-snake-dark.svg" />
  <img src="https://raw.githubusercontent.com/wanghaorui413-stack/wanghaorui413-stack/gh-pages/github-contribution-grid-snake.svg" width="100%" alt="贡献图贪吃蛇动画" />
</picture>

## 📫 联系我

如果你在招 AI 产品经理，或者想聊聊上面哪个项目，欢迎发邮件给我：**[13701813467@163.com](mailto:13701813467@163.com)**

<img src="assets/footer.svg" width="100%" alt="" />
