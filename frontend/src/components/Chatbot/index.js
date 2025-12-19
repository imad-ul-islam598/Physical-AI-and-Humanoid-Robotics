import React, { useState, useRef, useEffect } from "react";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import styles from "./index.module.css";
import useBaseUrl from "@docusaurus/useBaseUrl";

export default function ChatBot() {
  const logo = useBaseUrl("/img/logo.png");
  const { siteConfig } = useDocusaurusContext();
  const { api_key_endpoint } = siteConfig.customFields;

  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: "bot",
      text: "Hi, I am the your Assistant. Ask me anything related to books.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const messagesEndRef = useRef(null);
  const eventSourceRef = useRef(null);

  const toggleChat = () => {
    setIsOpen((prev) => !prev);
    if (!isOpen && eventSourceRef.current) {
      eventSourceRef.current.close();
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = { role: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    const streamingMessage = {
      role: "bot",
      text: "",
      isStreaming: true,
    };
    setMessages((prev) => [...prev, streamingMessage]);

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    try {
      const apiUrl = api_key_endpoint;

      eventSourceRef.current = new EventSource(
        `${apiUrl}?message=${encodeURIComponent(userMessage.text)}`
      );

      let accumulatedText = "";

      eventSourceRef.current.onmessage = (event) => {
        if (event.data === "[DONE]") {
          eventSourceRef.current?.close();
          setLoading(false);

          setMessages((prev) => {
            const newMessages = [...prev];
            const lastIndex = newMessages.length - 1;
            newMessages[lastIndex] = {
              ...newMessages[lastIndex],
              text: accumulatedText,
              isStreaming: false,
            };
            return newMessages;
          });
        } else {
          accumulatedText += event.data + " ";

          setMessages((prev) => {
            const newMessages = [...prev];
            const lastIndex = newMessages.length - 1;
            newMessages[lastIndex] = {
              ...newMessages[lastIndex],
              text: accumulatedText,
            };
            return newMessages;
          });
        }
      };

      eventSourceRef.current.onerror = () => {
        eventSourceRef.current?.close();
        setLoading(false);

        setMessages((prev) => {
          const newMessages = [...prev];
          const lastIndex = newMessages.length - 1;

          if (newMessages[lastIndex].text.trim() === "") {
            newMessages[lastIndex] = {
              ...newMessages[lastIndex],
              text: "Sorry, I encountered an error. Please try again.",
              isStreaming: false,
            };
          } else {
            newMessages[lastIndex].isStreaming = false;
          }
          return newMessages;
        });
      };
    } catch {
      eventSourceRef.current?.close();
      setLoading(false);

      setMessages((prev) => {
        const newMessages = [...prev];
        const lastIndex = newMessages.length - 1;
        newMessages[lastIndex] = {
          ...newMessages[lastIndex],
          text: "Sorry, I could not connect to the Neuro Library server.",
          isStreaming: false,
        };
        return newMessages;
      });
    }
  };

  return (
    <>
      <div className={styles.chatToggle} onClick={toggleChat}>
        {logo ? (
          <img src={logo} alt="Assistant" className={styles.chatIcon} />
        ) : (
          <div className={styles.chatIcon}>🤖</div>
        )}
        {!isOpen && (
          <div className={styles.tooltip}>
            Try Assistant
            <div className={styles.tooltiparrow}></div>
          </div>
        )}
      </div>

      {isOpen && (
        <div className={styles.chatPanel}>
          <div className={styles.chatHeader}>
            <span style={{ marginRight: "8px" }}>🤖</span>
            Chat Assistant
            <button className={styles.closeButton} onClick={toggleChat}>
              ✕
            </button>
          </div>

          <div className={styles.chatMessages}>
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={
                  msg.role === "user"
                    ? styles.userMessage
                    : `${styles.botMessage} ${
                        msg.isStreaming ? styles.streaming : ""
                      }`
                }
              >
                {msg.text}
                {msg.isStreaming && (
                  <span className={styles.streamingCursor}>▋</span>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          <form className={styles.chatForm} onSubmit={sendMessage}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask from the docs..."
              disabled={loading}
              autoFocus
            />
            <button type="submit" disabled={loading || !input.trim()}>
              {loading ? "..." : "Send"}
            </button>
          </form>
        </div>
      )}
    </>
  );
}
