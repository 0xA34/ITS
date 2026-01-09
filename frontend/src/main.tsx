import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";
import AppGate from "./AppGate";
import React from "react";

createRoot(document.getElementById("root")!).render(
<React.StrictMode>
    <AppGate>
        <App />
    </AppGate>
</React.StrictMode>
);
