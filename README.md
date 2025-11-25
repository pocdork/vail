# VAIL  
### Vulnerable Artificial Intelligence Lab

VAIL is a fully local, intentionally insecure AI testing environment designed for:

- Red-teaming practice  
- LLM security testing  
- Prompt injection research  
- Web app + AI vulnerability demonstrations  
- Training and learning AI attack scenarios  

It is **NOT a production-grade system**.  
Everything here is built insecure **on purpose**, for educational use only.

---

## ⚠️ Legal & Ethical Disclaimer

VAIL is created solely for **learning, security research, and safe defensive testing**.  

- Do **NOT** deploy this on the public internet.  
- Do **NOT** use these techniques against systems you do not own or have permission to test.  

By using this project, you accept full responsibility for its usage.

---

## 🚀 Features

VAIL ships with multiple realistic simulated AI vulnerabilities:

### ✔ HTML Injection  
User input is rendered directly into HTML without escaping.

### ✔ Reflected & Stored XSS  
Prompts and model responses are stored and later rendered unescaped.

### ✔ Role Manipulation  
A deliberately insecure endpoint allows changing user roles without authorization.

### ✔ Simulated Remote Command Injection  
User input is returned back as if executed (execution is **simulated**, not real — no system damage).

### ✔ Compliance Violations  
Sensitive data (prompts & commands) are logged in plain text.

### ✔ Optional Rate Limiting  
Disabled by default — can be enabled to demonstrate abuse.

### ✔ Jailbreak / Prompt Override  
If attackers include `OVERRIDE:`, the AI will follow that directive without restriction.

These vulnerabilities can be found and tested using tools such as:

- Strix
- Burp Suite
- OWASP ZAP
- Custom scripts  
- Manual penetration testing

---

## 🧠 How It Works

VAIL simulates a small AI platform that:

- Accepts a text prompt  
- Runs a local model (or simply echoes input if no model is installed)  
- Displays results directly in the browser **without sanitization**
- Stores logs insecurely for later retrieval  

It is a realistic environment for practicing common AI failures.

---

## 📦 Installation

### 1️⃣ Clone the repo

```bash
git clone https://github.com/you/vail.git
cd vail
```
### 2️⃣ Install dependencies

```bash
pip install fastapi uvicorn pydantic
```
### 3️⃣ Run the server
```bash
uvicorn sandbox:app --host 0.0.0.0 --port 8000
```
### Your local vulnerable AI lab is now running at:

```bash
http://127.0.0.1:8000
```
