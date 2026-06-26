import asyncio, json, numpy as np, yfinance as yf
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class DQN:
    def __init__(self, n_actions=3, n_features=4, lr=0.01, gamma=0.9, epsilon=0.9):
        self.n_actions, self.n_features = n_actions, n_features
        self.lr, self.gamma, self.epsilon = lr, gamma, epsilon
        self.q_table = np.zeros((10, 10, 10, 10, n_actions))
    
    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.argmax(self.q_table[state])
        return np.random.randint(0, self.n_actions)
    
    def learn(self, s, a, r, s_):
        q_predict = self.q_table[s][a]
        q_target = r + self.gamma * np.max(self.q_table[s_])
        self.q_table[s][a] += self.lr * (q_target - q_predict)

agents = [DQN() for _ in range(60)]
data = yf.download('BTC-USD', period='1d', interval='1m', progress=False)['Close'].values
data = (data - data.min()) / (data.max() - data.min() + 1e-8)
prices = (data * 9).astype(int)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    step = 0
    while True:
        states = []
        for i, agent in enumerate(agents):
            s = tuple(prices[step:step+4]) if step+4 < len(prices) else (0,0,0,0)
            a = agent.choose_action(s)
            s_ = tuple(prices[step+1:step+5]) if step+5 < len(prices) else (0,0,0,0)
            r = (prices[step+4] - prices[step+3]) * (a-1) if step+4 < len(prices) else 0
            agent.learn(s, a, r, s_)
            states.append({"id": i, "action": int(a), "q": float(np.max(agent.q_table[s]))})
        
        await websocket.send_json({"step": step, "price": int(prices[step]), "agents": states})
        step = (step + 1) % (len(prices) - 5)
        await asyncio.sleep(0.5)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)