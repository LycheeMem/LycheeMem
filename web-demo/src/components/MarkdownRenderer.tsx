import type { Components } from "react-markdown";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MarkdownRendererProps {
  content: string;
  /** 是否处于流式输出状态（显示光标） */
  streaming?: boolean;
}

const components: Components = {
  // 代码块：区分行内 code 和 fenced code block
  code({ className, children, ...props }) {
    const isBlock = className?.startsWith("language-");
    const lang = className?.replace("language-", "") ?? "";
    if (isBlock) {
      return (
        <div className="md-code-block">
          {lang && <div className="md-code-lang">{lang}</div>}
          <pre>
            <code className={className} {...props}>
              {children}
            </code>
          </pre>
        </div>
      );
    }
    return (
      <code className="md-inline-code" {...props}>
        {children}
      </code>
    );
  },

  // 链接：在新标签页打开，阻止 XSS
  a({ href, children }) {
    const safe =
      href && (href.startsWith("http://") || href.startsWith("https://") || href.startsWith("/"))
        ? href
        : "#";
    return (
      <a href={safe} target="_blank" rel="noopener noreferrer">
        {children}
      </a>
    );
  },

  // 表格
  table({ children }) {
    return (
      <div className="md-table-wrapper">
        <table>{children}</table>
      </div>
    );
  },

  // 引用块
  blockquote({ children }) {
    return <blockquote className="md-blockquote">{children}</blockquote>;
  },
};

export default function MarkdownRenderer({ content, streaming }: MarkdownRendererProps) {
  return (
    <div className={`md-body${streaming ? " md-streaming" : ""}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {content}
      </ReactMarkdown>
      {streaming && <span className="md-cursor" aria-hidden="true" />}
    </div>
  );
}
