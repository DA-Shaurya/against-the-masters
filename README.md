# ♟️ Against the Masters — Neural & Classical Dual-Engine Web Chess

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![JavaScript](https://img.shields.io/badge/Language-JavaScript%20ES6+-orange.svg)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![ONNX Runtime Web](https://img.shields.io/badge/ONNX%20Runtime-WebAssembly-blue.svg)](https://onnxruntime.ai/)
[![Web Audio API](https://img.shields.io/badge/Web%20Audio-Synthesized-green.svg)](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
[![Client-Side Execution](https://img.shields.io/badge/Execution-100%25%20In--Browser-brightgreen.svg)]()

> **Against the Masters** is a zero-dependency, presentation-grade web chess platform powered by ten neural networks trained on historical World Chess Champions' recorded games. It features a dual-engine architecture (**Classical Alpha-Beta with Champion Policy Bias** vs **Pure Neural Monte Carlo Tree Search**), real-time evaluation bars, endgame driving heuristics, and opening book interception — running 100% client-side in the browser.

---

## 🌟 Key Features

### 🧠 Dual Engine Search Architectures
- **Classical Alpha-Beta Search**: Combines material evaluation, Piece-Square Tables (PST), tactical threat protection (penalizing hanging major pieces under pawn attack by over 600 centipawns), and Quiescence search with imitation-trained **Champion Policy Bias** (+0 to +15 centipawn boost for signature champion moves).
- **Neural MCTS Engine**: Executes pure Monte Carlo Tree Search using a dual-head Policy/Value network (`move_logits` and `value` heads), leaf expansion, PUCT move selection ($C_{\text{PUCT}} = 1.5$), exact terminal state resolution (Checkmate = -1.0, Draw = 0.0), and alternating sign backpropagation.

### 🛡️ Material-Anchored Horizon Blending
To prevent Monte Carlo Tree Search from making un-calculated piece sacrifices when search depth cuts off, the MCTS engine blends neural evaluations with static material values:
$$V_{\text{final}}(s) = 0.8 \cdot V_{\text{neural}}(s) + 0.2 \cdot \tanh\left(\frac{\text{StaticScore}(s)}{400}\right)$$

### 👑 Endgame King-Driving Heuristics
When total non-pawn material drops below 1,500, the evaluation function dynamically calculates:
1. **Losing King Edge Penalty**: Measures how far the losing King is pushed away from center squares (d4, d5, e4, e5), awarding +10 to +40 centipawns to the winning side.
2. **King Proximity Bonus**: Measures Manhattan distance between the winning King and losing King, granting +10 to +30 centipawns when the winning King closes in to enforce checkmate.

### 📊 Real-Time Dynamic Evaluation Bar
- Slim, elegant vertical evaluation bar embedded directly inside the board frame.
- Calculates win probability percentage (0% to 100%) and smoothly transitions using CSS easing (`transition: height 0.5s ease;`).
- Updates immediately after both AI moves and player turns.

### 📖 Grandmaster Opening Book Interception
- Intercepts opening positions against `opening_book.json` to play theoretical grandmaster moves instantly (0 ms) with visual indicators (`📖 Book Move`).

### 🎵 Synthesized Web Audio API
- Fully custom Web Audio API synthesizer for sound effects (move, capture, check, and game over) with zero external media files.

### 📈 Post-Game Style Matcher Report
- Evaluates played moves against all 10 World Champion policy distributions to display a post-game style breakdown (e.g. *78% Garry Kasparov, 16% Anatoly Karpov, 6% Mikhail Tal*).

---

## 🏛️ The Hall of Champions

The platform allows you to play against 10 distinct World Chess Champions, each modeled by policy networks trained on their historical PGN game collections:

| Champion | Reign | Playstyle Focus |
| :--- | :---: | :--- |
| **Emanuel Lasker** | 1894–1921 | Psychological play, stubborn defense, pragmatic counter-attack |
| **José Raúl Capablanca** | 1921–1927 | Crystal-clear positional mastery, endgame precision |
| **Alexander Alekhine** | 1927–1946 | Aggressive tactical combinations, relentless initiative |
| **Mikhail Botvinnik** | 1948–1963 | Methodical strategy, deep research, systematic planning |
| **Mikhail Tal** | 1960–1961 | Explosive intuitive sacrifices, sharp tactical chaos |
| **Bobby Fischer** | 1972–1975 | Relentless accuracy, opening preparation, energetic play |
| **Anatoly Karpov** | 1975–1985 | Prophylactic squeeze, positional restriction |
| **Garry Kasparov** | 1985–2000 | Dynamic initiative, heavy tactical pressure, deep calculation |
| **Viswanathan Anand** | 2000–2013 | Lightning calculation, intuitive speed, versatile tactics |
| **Magnus Carlsen** | 2013–2023 | Universal flexibility, grinding endgame technique |

---

## 🔬 Mathematical & Technical Architecture

### 1. Board Tensor Representation
Chess board states are converted into a $[1, 17, 8, 8]$ binary tensor:
- **Planes 0–5**: White pieces (P, N, B, R, Q, K)
- **Planes 6–11**: Black pieces (P, N, B, R, Q, K)
- **Plane 12**: Active turn indicator (1 for White, 0 for Black)
- **Planes 13–16**: Castling rights (K, Q, k, q)

### 2. PUCT Move Selection Formula
MCTS node selection during tree traversal is governed by:
$$U(s, a) = -Q(s, a) + C_{\text{PUCT}} \cdot P(s, a) \cdot \frac{\sqrt{N(\text{parent})}}{1 + N(\text{child})}$$
where $C_{\text{PUCT}} = 1.5$ balances exploration against high-prior policy moves $P(s, a)$.

---

## 📁 Repository Structure

```
against-the-masters/
├── index.html                  # Primary Web Application Entry Point
├── against_the_masters.html    # Standalone Application Target
├── against_the_masters_3.html  # Application Target Sync
├── engine-worker.js            # Web Worker Engine Script
├── unified_dual_head.onnx      # Preloaded Dual-Head Policy/Value ONNX Model
├── opening_book.json           # Grandmaster Opening Database
├── chess.min.js                # Chess Logic Engine (v0.10.3)
├── ort.min.js                  # ONNX Runtime Web (v1.23.2)
├── *.pgn                       # Master PGN Databases (Lasker, Fischer, etc.)
├── LICENSE                     # MIT License
└── README.md                   # Technical Documentation & User Guide
```

---

## 🚀 Quick Start Guide

### Option 1: Direct Local Execution
Because `against-the-masters` uses local `ort.min.js` and `chess.min.js` files, you can launch a local HTTP server using Python:

```bash
# Clone the repository
git clone https://github.com/DA-Shaurya/against-the-masters.git
cd against-the-masters

# Start a local HTTP server
python -m http.server 8000
```
Open your browser and navigate to **`http://localhost:8000`**.

### Option 2: Live Web Deployment
You can deploy this repository instantly via **GitHub Pages**, **Vercel**, or **Netlify** with zero build step requirements (static HTML/JS/ONNX).

---

## 🤝 Contributing

Contributions, bug reports, and engine enhancements are welcome! Feel free to open an Issue or submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more details.

---

## 👤 Author

**Shaurya Singh**
- GitHub: [@DA-Shaurya](https://github.com/DA-Shaurya)
