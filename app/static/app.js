const log = document.querySelector("#log");
const citations = document.querySelector("#citations");
const message = document.querySelector("#message");
const form = document.querySelector("#composer");
const confirmBox = document.querySelector("#confirm");
const loginPanel = document.querySelector("#login-panel");
const loginForm = document.querySelector("#login-form");
const loginError = document.querySelector("#login-error");
const workspace = document.querySelector("#workspace");
const sessionBar = document.querySelector("#session-bar");
const who = document.querySelector("#who");
const chatList = document.querySelector("#chat-list");
let pending = null;
let demosLoaded = false;
let actorId = null;
let store = { activeId: null, chats: [] };

function storageKey() {
  return `innovatech-chats:${actorId || "guest"}`;
}

function activeChat() {
  return store.chats.find((chat) => chat.id === store.activeId) || store.chats[0];
}

function loadStore() {
  try {
    store = JSON.parse(localStorage.getItem(storageKey())) || { activeId: null, chats: [] };
  } catch (error) {
    store = { activeId: null, chats: [] };
  }
  if (!Array.isArray(store.chats)) store.chats = [];
  if (!store.chats.length) {
    const chat = { id: `c-${Date.now()}`, title: "New chat", messages: [] };
    store.chats.push(chat);
    store.activeId = chat.id;
  }
  if (!store.chats.some((chat) => chat.id === store.activeId)) store.activeId = store.chats[0].id;
  localStorage.setItem(storageKey(), JSON.stringify(store));
}

function saveStore() {
  localStorage.setItem(storageKey(), JSON.stringify(store));
  renderChatList();
}

function renderChatList() {
  chatList.replaceChildren();
  store.chats.forEach((chat) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = chat.id === store.activeId ? "chat-open active" : "chat-open";
    button.textContent = chat.title || "New chat";
    button.addEventListener("click", () => openChat(chat.id));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "chat-delete secondary";
    remove.textContent = "Delete";
    remove.setAttribute("aria-label", `Delete ${chat.title || "New chat"}`);
    remove.addEventListener("click", () => deleteChat(chat.id));
    item.append(button, remove);
    chatList.append(item);
  });
}

function paintBubble(entry, nextSteps) {
  const node = document.createElement("article");
  node.className = `bubble ${entry.role}`;
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = entry.extra || (entry.role === "user" ? "You" : "Assistant");
  const body = document.createElement("div");
  body.className = "text";
  body.textContent = entry.text;
  node.append(meta, body);
  if (nextSteps && nextSteps.length) {
    const choices = document.createElement("div");
    choices.className = "next-steps";
    nextSteps.forEach((step) => {
      const choice = document.createElement("button");
      choice.type = "button";
      choice.className = "secondary";
      choice.textContent = step.label;
      choice.addEventListener("click", () => ask(step.label, { selectedStep: step }));
      choices.append(choice);
    });
    node.append(choices);
  }
  log.append(node);
}

function renderActive() {
  const chat = activeChat();
  log.replaceChildren();
  const lastAssistant = chat.messages.findLastIndex((entry) => entry.role === "assistant");
  chat.messages.forEach((entry, index) => {
    paintBubble(entry, index === lastAssistant ? entry.nextSteps : null);
  });
  log.scrollTop = log.scrollHeight;
  const latest = lastAssistant >= 0 ? chat.messages[lastAssistant] : null;
  renderCitations(latest ? latest.citations : null);
  pending = latest && latest.pending ? latest.pending : null;
  confirmBox.classList.toggle("hidden", !pending);
  renderChatList();
}

function remember(entry) {
  const chat = activeChat();
  chat.messages.push(entry);
  if (entry.role === "user" && (!chat.title || chat.title === "New chat")) {
    chat.title = entry.text.length > 48 ? `${entry.text.slice(0, 48)}…` : entry.text;
  }
  saveStore();
}

function openChat(id) {
  store.activeId = id;
  saveStore();
  renderActive();
}

function deleteChat(id) {
  store.chats = store.chats.filter((chat) => chat.id !== id);
  if (!store.chats.length) {
    store.chats.push({ id: `c-${Date.now()}`, title: "New chat", messages: [] });
  }
  if (!store.chats.some((chat) => chat.id === store.activeId)) {
    store.activeId = store.chats[0].id;
  }
  pending = null;
  confirmBox.classList.add("hidden");
  saveStore();
  renderActive();
}

function startNewChat() {
  const current = activeChat();
  if (current && current.messages.length === 0) return;
  const chat = { id: `c-${Date.now()}`, title: "New chat", messages: [] };
  store.chats.unshift(chat);
  store.activeId = chat.id;
  pending = null;
  confirmBox.classList.add("hidden");
  saveStore();
  renderActive();
}

function renderCitations(items) {
  citations.replaceChildren();
  if (!items || !items.length) {
    citations.textContent = "None yet.";
    return;
  }
  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "cite";
    const title = document.createElement("h3");
    title.textContent = `${item.document_id} · ${item.section}`;
    const body = document.createElement("pre");
    body.textContent = item.snippet || "";
    card.append(title, body);
    citations.append(card);
  });
}

function showSignedOut() {
  loginPanel.classList.remove("hidden");
  workspace.classList.add("hidden");
  sessionBar.classList.add("hidden");
  who.textContent = "";
  actorId = null;
  pending = null;
  confirmBox.classList.add("hidden");
  log.replaceChildren();
}

function showSignedIn(profile) {
  loginPanel.classList.add("hidden");
  workspace.classList.remove("hidden");
  sessionBar.classList.remove("hidden");
  who.textContent = `${profile.full_name} · ${profile.employee_id}`;
  loginError.textContent = "";
  actorId = profile.employee_id;
  loadStore();
  renderActive();
  if (!demosLoaded) loadDemos();
}

async function ask(text, options = {}) {
  const button = form.querySelector("button");
  button.disabled = true;
  if (!options.silentUser) remember({ role: "user", text });
  renderActive();
  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        confirm_action: Boolean(options.confirm),
        pending_action: options.pending || null,
        selected_step: options.selectedStep || null,
      }),
    });
    if (response.status === 401) {
      showSignedOut();
      loginError.textContent = "Sign in again to continue.";
      return;
    }
    const data = await response.json();
    remember({
      role: "assistant",
      text: data.answer || "No answer returned.",
      extra: data.intent ? `Assistant · ${data.intent}` : "Assistant",
      nextSteps: data.next_steps || [],
      trace: data.trace || [],
      citations: data.citations || [],
      pending: data.pending_action || null,
    });
    renderActive();
  } catch (error) {
    remember({
      role: "assistant",
      text: "The request failed before the agent could answer. Nothing was changed.",
      extra: "Assistant",
      nextSteps: [],
      trace: [],
      citations: [],
      pending: null,
    });
    renderActive();
  } finally {
    button.disabled = false;
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = message.value.trim();
  if (!text) return;
  message.value = "";
  ask(text);
});

loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  loginError.textContent = "";
  const response = await fetch("/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      employee_id: document.querySelector("#login-id").value.trim(),
      password: document.querySelector("#login-password").value,
    }),
  });
  if (!response.ok) {
    loginError.textContent = "Employee id or password is incorrect.";
    return;
  }
  document.querySelector("#login-password").value = "";
  showSignedIn(await response.json());
});

document.querySelector("#logout").addEventListener("click", async () => {
  await fetch("/logout", { method: "POST" });
  citations.textContent = "None yet.";
  showSignedOut();
});

document.querySelector("#new-chat").addEventListener("click", startNewChat);

document.querySelector("#confirm-yes").addEventListener("click", () => {
  if (!pending) return;
  const action = pending;
  confirmBox.classList.add("hidden");
  pending = null;
  ask("Yes, store the mock ticket.", { confirm: true, pending: action });
});

document.querySelector("#confirm-no").addEventListener("click", () => {
  confirmBox.classList.add("hidden");
  pending = null;
  const chat = activeChat();
  const latest = chat.messages.findLast((entry) => entry.role === "assistant");
  if (latest) latest.pending = null;
  remember({
    role: "assistant",
    text: "The mock ticket was discarded. Nothing was stored.",
    extra: "Assistant · cancelled",
    nextSteps: [],
    trace: [],
    citations: [],
    pending: null,
  });
  renderActive();
});

function loadDemos() {
  demosLoaded = true;
  fetch("/api/demo-tasks")
    .then((response) => response.json())
    .then((data) => {
      const host = document.querySelector("#demos");
      data.tasks.forEach((task) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = task.title;
        button.addEventListener("click", () => ask(task.message));
        host.append(button);
      });
    });
}

fetch("/api/me").then(async (response) => {
  if (response.ok) showSignedIn(await response.json());
  else showSignedOut();
});
