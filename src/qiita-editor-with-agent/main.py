"""
- [x] ブラウザで動作する UI をもつ
- [ ] 人間が執筆した記事を受け付ける画面を持つ
    - [x] UI で記事を編集できるようにする
    - [x] Markdown をレンダリングしたプレビューを表示する
    - [ ] 外部エディタで編集できるようにファイルから読み取る機能もほしい
- [x] 人間から審査プロセス開始の合図を受ける
- [x] 記事のチェックが通ったら人間に承認を依頼する
- [ ] 人間が承認しなかったら動作を停止する
- [x] 人間の承認は問わずに媒体に投稿する
- [x] 人間が承認したら媒体に投稿する
    - [ ] 公開モード：媒体に投稿してすぐに公開する
    - [x] 非公開モード：媒体に投稿するけど公開しない
- [x] 投稿したら投稿結果を表示する
    - [x] 結果の例：正常異常のステータス、URL

"""

import json
import os
from typing import Any, List, Literal, Tuple

import gradio as gr
from agents import Agent, RunContextWrapper, Runner, function_tool
from pydantic import BaseModel
from qiita import Qiita
from qiita.v2.models.create_item_request import CreateItemRequest
from qiita.v2.models.item_tag import ItemTag

EDITOR_LINES = os.getenv("EDITOR_LINES", 30)
PREVIEW_HEIGHT = os.getenv("PREVIEW_HEIGHT", 700)
MAX_ARTICLE_TAGS = os.getenv("MAX_ARTICLE_TAGS", 5)


class ArticleInformation(BaseModel):
    """
    OpenAI Agents SDK の Runner 内部で利用する記事の情報
    """

    body: str = ""  # 記事の Body
    tags: List[ItemTag]  # タグ(5個まで)
    title: str = ""  # 記事のタイトル
    private: bool = True  # 非公開記事
    tweet: bool = False  # x.com にポストするか？
    slide: bool = False  # スライドモード Off


class CheckResult(BaseModel):
    """
    編集者とチェック者が返すメッセージフォーマット
    """

    status: Literal["reject", "accept"]
    name: str
    comment: str


@function_tool
async def confirm_to_human(
    run_ctx: RunContextWrapper[Any], message: str
) -> str:
    """
    システムから人間へ問い合わせを行い結果を取得する関数

    Args:
        message: システムから人間への質問の文字列
    """

    return "accept"  # 何でも許可するザルモード


@function_tool
def publish_to_platform(run_ctx: RunContextWrapper[Any]) -> str:
    """
    審査済みの記事を媒体へ投稿する関数
    """
    article_info: ArticleInformation = run_ctx.context
    qiita = Qiita(access_token=os.environ["QIITA_API_ACCESS_TOKEN"])
    response = qiita.create_item_with_http_info(
        CreateItemRequest(**article_info.model_dump())
    )
    return response.data


# 公開責任者エージェント
applover_agent = Agent(
    name="公開責任者",
    instructions="""
    あなたは記事を媒体に投稿する前に公開の可否を最終判断する公開責任者です。
    毎回、confirm_to_human 関数を使って第三者に判断を仰いでください。
    第三者が accept と判断した場合は、publish_to_platform 関数を呼び出して
    記事を投稿してください。
    第三者が reject と判断した場合は、reject した旨を応答してください。
    ユーザーに対しては判断結果と投稿したURLを返してください。
    """,
    tools=[
        confirm_to_human,
        publish_to_platform,
    ],
)

# 読みやすさチェックエージェント(Tool化)
readability_check_tool = Agent(
    name="読みやすさをチェックするエージェント",
    instructions="""
    記事が読者にとって読みやすいかどうかを判定してください。
    結果は reject または accept とします。
    comment に判定した理由を簡潔に記載してください。
    """,
    output_type=CheckResult,
).as_tool(
    tool_name="readablility_check_tool",
    tool_description="記事が読者にとって読みやすいかどうかを判定します",
)

# 内容の質と正確性をチェックするエージェント(Tool化)
quailty_check_tool = Agent(
    name="内容の質と正確性をチェックするエージェント",
    instructions="""
    技術的な正確さ、情報の鮮度、独自性、目的の明確さについて
    審査して、読者にとって有用かどうかを判定してください。
    結果は reject または accept とします。
    comment に判定した理由を簡潔に記載してください。
    """,
    output_type=CheckResult,
).as_tool(
    tool_name="quality_check_tool",
    tool_description="記事の質と正確性を判定します",
)

# コードブロックの適切さをチェックするエージェント(Tool化)
code_format_check_tool = Agent(
    name="コードのフォーマットをチェックするエージェント",
    instructions="""
    記事の中のコードブロックについて、コードのフォーマットや
    読みやすさを審査して、読者にとって読みやすいかを判定して
    ください。
    結果は reject または accept とします。
    comment に判定した理由を簡潔に記載してください。
    """,
    output_type=CheckResult,
).as_tool(
    tool_name="code_format_check_tool",
    tool_description="コードのフォーマットや読みやすさを判定します",
)


# 編集者エージェント
editor_agent = Agent(
    name="編集者",
    instructions="""
    あなたは編集者です。
    執筆者から受け取った記事を複数のチェック者の意見を聞いてチェックします。
    すべてのチェック者が問題なしと回答した場合は、公開責任者へ公開の判断と
    媒体への投稿作業を依頼してください。
    一人でもチェック者から問題ありと回答を得た場合は、執筆者へ公開できない
    理由を返答してください。
    """,
    handoffs=[applover_agent],
    tools=[
        readability_check_tool,
        quailty_check_tool,
        code_format_check_tool,
    ],
)


async def review_and_post(title, tags, body, chat_history):
    """
    記事のレビュー、投稿確認、投稿作業を行う。結果をチャットの履歴で返す。
    """
    chat_history.append({"role": "user", "content": f"「{title}」を審査して"})
    yield chat_history

    def input_validation_check(
        title: str, tags: str, body: str
    ) -> Tuple[bool, str]:
        """入力チェック"""
        if len(title) == 0:
            return False, "タイトルが空です"
        if len(body) == 0:
            return False, "本文が空です"
        if len(tags.strip().split(" ")) > MAX_ARTICLE_TAGS:
            return False, "タグが多すぎます。5 以下にしてください。"
        return True, "問題ありません"

    # 入力チェック
    valid_flag, message = input_validation_check(title, tags, body)
    if not valid_flag:
        # 入力チェックエラー
        chat_history.append({"role": "assistant", "content": message})
        yield chat_history
    else:
        # 記事の情報をインスタンス化
        article_info = ArticleInformation(
            title=str(title).strip(),
            tags=[
                ItemTag(name=x, versions=[])
                for x in str(tags).strip().split(" ")
            ],
            body=body,
        )

        # エージェントを実行
        response = await Runner.run(
            editor_agent,
            input=json.dumps(
                {
                    "title": title,
                    "tags": tags,
                    "body": body,
                },
                ensure_ascii=False,
            ),
            context=article_info,
        )

        # エージェントの出力をチャットに追加
        chat_history.append(
            {"role": "assistant", "content": response.final_output}
        )

        yield chat_history


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            title_pane = gr.Textbox(lines=1, show_label=False)
            tags_pane = gr.Textbox(lines=1, show_label=False)
            editor_pane = gr.Textbox(lines=EDITOR_LINES, show_label=False)
        preview_pane = gr.Markdown(height=PREVIEW_HEIGHT)
        chat_history_pane = gr.Chatbot(type="messages", height=PREVIEW_HEIGHT)
    submit = gr.Button("チェックお願いします！")

    # markdown プレビューの更新
    gr.on(
        [editor_pane.change],
        fn=lambda x: gr.Markdown(x),
        inputs=[editor_pane],
        outputs=[preview_pane],
    )

    # チェック&投稿依頼
    gr.on(
        [submit.click],
        fn=review_and_post,
        inputs=[title_pane, tags_pane, editor_pane, chat_history_pane],
        outputs=[chat_history_pane],
    )


if __name__ == "__main__":
    demo.launch(share=False, debug=True)
