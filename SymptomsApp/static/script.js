(function () {
  const chatbotToggler = document.querySelector(".chatbot-toggler");
  const closeBtn = document.querySelector(".close-btn");
  const chatbox = document.querySelector(".chatbox");
  const chatInput = document.querySelector(".chat-input textarea");
  const sendChatBtn = document.querySelector(".chat-input span");
  const langSelectNav = document.getElementById("language-select");
  const langSelectChat = document.getElementById("chat-lang-select");
  const welcomeMsgElem = document.getElementById("welcome-msg");
  const promptChips = document.querySelectorAll(".prompt-chip");

  let currentLang = "en";
  let userMessage = null;
  const inputInitHeight = 48;

  const WELCOME_MESSAGES = {
    en: "Hi there! 👋\n\nI am SymptoTrackAI, your intelligent symptom monitoring assistant.\n\nHow are you feeling today? Please describe your symptoms (e.g. fever, headache, cough, rash).",
    hi: "नमस्ते! 👋\n\nमैं SymptoTrackAI हूँ, आपका स्वास्थ्य एवं लक्षण निगरानी सहायक।\n\nआप आज कैसा महसूस कर रहे हैं? कृपया अपने लक्षणों का वर्णन करें (जैसे: बुखार, सिरदर्द, खांसी, दाने)।",
    te: "నమస్కారం! 👋\n\nనేను SymptoTrackAI, మీ ఆరోగ్య మరియు లక్షణాల పర్యవేక్షణ సహాయకుడిని.\n\nఈరోజు మీరు ఎలా ఉన్నారు? దయచేసి మీ లక్షణాలను వివరించండి (ఉదా: జ్వరం, తలనొప్పి, దగ్గు, దద్దుర్లు).",
    es: "¡Hola! 👋\n\nSoy SymptoTrackAI, tu asistente de monitoreo de síntomas.\n\n¿Cómo te sientes hoy? Por favor describe tus síntomas (ej. fiebre, dolor de cabeza, tos, sarpullido)."
  };

  const setLanguage = (lang) => {
    currentLang = lang;
    if (langSelectNav) langSelectNav.value = lang;
    if (langSelectChat) langSelectChat.value = lang;
    if (welcomeMsgElem && WELCOME_MESSAGES[lang]) {
      welcomeMsgElem.textContent = WELCOME_MESSAGES[lang];
    }
  };

  if (langSelectNav) {
    langSelectNav.addEventListener("change", (e) => setLanguage(e.target.value));
  }
  if (langSelectChat) {
    langSelectChat.addEventListener("change", (e) => setLanguage(e.target.value));
  }

  promptChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const query = chip.getAttribute("data-query");
      if (query && chatInput) {
        chatInput.value = query;
        document.body.classList.add("show-chatbot");
        handleChat();
      }
    });
  });

  const createChatLi = (message, className) => {
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", className);
    let chatContent =
      className === "outgoing"
        ? `<p></p>`
        : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
    chatLi.innerHTML = chatContent;
    chatLi.querySelector("p").textContent = message;
    return chatLi;
  };

  const generateResponse = async (chatElement) => {
    const messageElement = chatElement.querySelector("p");

    try {
      const params = new URLSearchParams({
        mytext: userMessage,
        lang: currentLang,
      });
      const apiUrl = `/ChatData?${params.toString()}`;

      const response = await fetch(apiUrl, { method: "GET" });
      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const data = await response.text();
      messageElement.textContent = data;
    } catch (error) {
      messageElement.textContent = `⚠️ Error retrieving response: ${error.message}. Please try again.`;
    } finally {
      chatbox.scrollTo(0, chatbox.scrollHeight);
    }
  };

  const handleChat = () => {
    if (!chatInput) return;
    userMessage = chatInput.value.trim();
    if (!userMessage) return;

    chatInput.value = "";
    chatInput.style.height = `${inputInitHeight}px`;

    chatbox.appendChild(createChatLi(userMessage, "outgoing"));
    chatbox.scrollTo(0, chatbox.scrollHeight);

    setTimeout(() => {
      const incomingChatLi = createChatLi("Analyzing symptoms...", "incoming");
      chatbox.appendChild(incomingChatLi);
      chatbox.scrollTo(0, chatbox.scrollHeight);
      generateResponse(incomingChatLi);
    }, 300);
  };

  if (chatInput) {
    chatInput.addEventListener("input", () => {
      chatInput.style.height = `${inputInitHeight}px`;
      chatInput.style.height = `${chatInput.scrollHeight}px`;
    });

    chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleChat();
      }
    });
  }

  if (sendChatBtn) {
    sendChatBtn.addEventListener("click", handleChat);
  }

  if (closeBtn) {
    closeBtn.addEventListener("click", () => {
      document.body.classList.remove("show-chatbot");
    });
  }

  if (chatbotToggler) {
    chatbotToggler.addEventListener("click", () => {
      document.body.classList.toggle("show-chatbot");
    });
  }
})();