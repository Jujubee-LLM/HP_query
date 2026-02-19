import { Suspense } from "react";
import QueryForm from "./components/QueryForm";

export default function Page() {
  return (
    <Suspense fallback={<div className="small">加载中…</div>}>
      <QueryForm />
    </Suspense>
  );
}
