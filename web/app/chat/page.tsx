import { Suspense } from "react";
import { ChatUI } from "@/components/chat/chat-ui";

export default function ChatPage() {
  return (
    <Suspense fallback={null}>
      <ChatUI />
    </Suspense>
  );
}
