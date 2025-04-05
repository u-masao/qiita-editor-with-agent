"""
# Qiita Editor with Agent

AI エージェントがブログの原稿をチェックして投稿してくれるエディタ

サポートする媒体: Qiita
"""

import json
import os
from typing import Any, List, Literal, Tuple

from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from agents import Agent, RunContextWrapper, Runner, function_tool
from pydantic import BaseModel
from qiita import Qiita
from qiita.v2.models.create_item_request import CreateItemRequest
from qiita.v2.models.item_tag import ItemTag

EDITOR_LINES = os.getenv("EDITOR_LINES", 30)
PREVIEW_HEIGHT = os.getenv("PREVIEW_HEIGHT", 600)
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
    編集者とチェック者が返すメッセージのフォーマット
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
def publish_to_platform(
    run_ctx: RunContextWrapper[Any], platform: Literal["qiita"]
) -> str:
    """
    審査済みの記事を媒体へ投稿する関数

    Args:
        platform: 投稿先の媒体名, qiita をサポート
    """

    if platform not in ["qiita"]:
        raise ValueError(f"その媒体はサポートしていません: {platform}")

    if platform == "qiita":
        # Qiita に投稿する処理
        article_info: ArticleInformation = run_ctx.context
        qiita = Qiita(access_token=os.environ["QIITA_API_ACCESS_TOKEN"])
        response = qiita.create_item_with_http_info(
            CreateItemRequest(**article_info.model_dump())
        )
        result = response.data

    return result


# 公開責任者エージェント
applover_agent = Agent(
    name="公開責任者",
    instructions="""
    あなたは記事を媒体に投稿する前に公開の可否を最終判断する公開責任者です。
    執筆者や編集者から受け取る記事を媒体に投稿すべきかを判断します。
    絶対にユーザー、執筆者、編集者へ意見を聞かないでください。

    必ず confirm_to_human 関数を使って第三者に判断を仰いでください。
    第三者が accept と判断した場合は publish_to_platform 関数を呼び出して
    記事を投稿してください。
    第三者が reject と判断した場合は reject した旨を応答してください。

    ユーザーに対しては判断結果、判断結果の根拠、投稿記事の URL を返してください。
    絶対にユーザー、執筆者、編集者へ意見を聞かないでください。
    """,
    tools=[
        confirm_to_human,  # 人間に最終確認するツール
        publish_to_platform,  # 媒体に投稿するツール
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
        readability_check_tool,  # 読みやすさをチェックするエージェント
        quailty_check_tool,  # 品質をチェックするエージェント
        code_format_check_tool,  # コードをチェックするエージェント
    ],
)


async def review_and_post(
    title, tags, body, private_flag, slide_flag, chat_history
):
    """
    記事のレビュー、投稿確認、投稿作業を行う。結果をチャットの履歴で返す。
    """
    chat_history += [
        {"role": "user", "content": f"「{title}」を Qiita に投稿して"},
        {"role": "assistant", "content": "はい、チェックしてから投稿しますね"},
        {"role": "assistant", "content": "・・・"},
    ]
    yield chat_history

    def input_validation_check(
        title: str, tags: str, body: str
    ) -> Tuple[bool, str]:
        """入力チェック"""
        if len(title) == 0:
            return False, "タイトルが空です。タイトルを入力してください。"
        if len(body) == 0:
            return False, "本文が空です。本文を入力してください。"
        if len(tags.strip().split(" ")) > MAX_ARTICLE_TAGS:
            return False, "タグが多すぎます。5 以下にしてください。"
        return True, "問題ありません。"

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
            private=private_flag,
            slide=slide_flag,
        )

        # エージェントを実行
        # 記事の情報と投稿先をプロンプトとして渡す(JSON形式)
        # 途中で LLM に編集されるのを避けるため記事の text は context で渡す
        response = await Runner.run(
            editor_agent,  # 編集者エージェントに依頼
            input=json.dumps(  # LLM には JSON 形式で情報を渡す
                {
                    "platform": "qiita",
                    "title": title,
                    "tags": tags,
                    "body": body,
                    "slide_flag": slide_flag,
                },
                ensure_ascii=False,  # 漢字等をエスケープさせない
            ),
            context=article_info,  # Tool に渡す情報
        )

        # エージェントの出力をチャットに追加
        chat_history.append(
            {"role": "assistant", "content": response.final_output}
        )

        yield chat_history


HEADER_TEXT = """
### AI エージェントが Qiita の原稿をチェックして投稿してくれるエディタ
"""


with gr.Blocks() as demo:
    # UI コンポーネントを配置
    preview_kwargs = {
        "height": PREVIEW_HEIGHT,
        "line_breaks": True,
        "show_label": False,
        "value": "# プレビュー\n\nここに本文のプレビューが表示されます",
    }
    gr.Markdown(HEADER_TEXT, show_label=False)
    with gr.Row():
        with gr.Column():
            with gr.Accordion("タイトル、公開設定", open=False):
                title_pane = gr.Textbox(lines=1, max_lines=1, label="タイトル")
                tags_pane = gr.Textbox(
                    lines=1, max_lines=1, label="タグ(5個まで)"
                )
                private_flag = gr.Checkbox(value=True, label="限定公開")
                slide_flag = gr.Checkbox(value=False, label="スライドモード")
            body_pane = gr.Textbox(
                lines=EDITOR_LINES,
                show_label=True,
                autofocus=True,
                label="本文",
            )
        with gr.Tab("メッセージ"):
            welcome_message = {
                "role": "assistant",
                "content": """
                原稿を入力したら「チェック＆投稿」ボタンを押してください。
                内容に問題がなければ Qiita に投稿します。
                """,
            }
            chat_history_pane = gr.Chatbot(
                [welcome_message],
                type="messages",
                height=PREVIEW_HEIGHT,
                show_label=False,
            )
            submit_button = gr.Button("チェック＆投稿")
        with gr.Tab("プレビュー"):
            preview_pane = gr.Markdown(**preview_kwargs)

    # markdown プレビューの更新
    gr.on(
        [body_pane.change],
        fn=lambda x: gr.Markdown(x, **preview_kwargs),
        inputs=[body_pane],
        outputs=[preview_pane],
    )

    # チェック&投稿依頼
    gr.on(
        [submit_button.click],
        fn=review_and_post,
        inputs=[
            title_pane,
            tags_pane,
            body_pane,
            private_flag,
            slide_flag,
            chat_history_pane,
        ],
        outputs=[chat_history_pane],
    )


if __name__ == "__main__":
    # Gradio UI を起動
    demo.launch(share=False, debug=True)
