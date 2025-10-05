"""
グラフ生成コンポーネント
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional


class GraphGenerator:
    """グラフ生成クラス"""
    
    def __init__(self):
        from ..config import config
        self.colors = px.colors.qualitative.Set3
        self.template = config.GRAPH_TEMPLATE
    
    def create_accuracy_by_category_bar(self, df: pd.DataFrame) -> go.Figure:
        """カテゴリ別識別精度の棒グラフ"""
        if df.empty:
            return self._create_empty_figure("データがありません")

        # M5Stack分類データを優先的に使用
        if 'm5stack_correct' in df.columns and not df['m5stack_correct'].isna().all():
            correct_col = 'm5stack_correct'
            title = "カテゴリ別M5Stack分類精度"
        elif 'dummy_identification_correct' in df.columns:
            correct_col = 'dummy_identification_correct'
            title = "カテゴリ別ダミー識別精度（参考）"
        elif 'identification_correct' in df.columns:
            # 旧形式との互換性
            correct_col = 'identification_correct'
            title = "カテゴリ別識別精度"
        else:
            return self._create_empty_figure("識別データがありません")

        # カテゴリ別精度を計算
        category_stats = df.groupby('category').agg({
            correct_col: ['mean', 'count']
        }).round(3)

        category_stats.columns = ['accuracy', 'count']
        category_stats = category_stats.reset_index()
        category_stats['accuracy_pct'] = category_stats['accuracy'] * 100
        
        fig = go.Figure(data=[
            go.Bar(
                x=category_stats['category'],
                y=category_stats['accuracy_pct'],
                text=category_stats['accuracy_pct'].round(1).astype(str) + '%',
                textposition='auto',
                name='識別精度',
                marker_color=self.colors[0]
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="音声カテゴリ",
            yaxis_title="識別精度 (%)",
            template=self.template,
            showlegend=False
        )
        
        return fig
    
    def create_confidence_histogram(self, df: pd.DataFrame) -> go.Figure:
        """信頼度分布のヒストグラム"""
        if df.empty:
            return self._create_empty_figure("データがありません")

        # M5Stackには信頼度がないため、dB値のヒストグラムを表示
        if 'm5stack_db_value' in df.columns and not df['m5stack_db_value'].isna().all():
            data_col = 'm5stack_db_value'
            title = "M5Stack検出dB値分布"
            xaxis_title = "dB値"
        elif 'dummy_confidence' in df.columns:
            data_col = 'dummy_confidence'
            title = "ダミー識別信頼度分布（参考）"
            xaxis_title = "信頼度"
        elif 'confidence' in df.columns:
            # 旧形式との互換性
            data_col = 'confidence'
            title = "識別信頼度分布"
            xaxis_title = "信頼度"
        else:
            return self._create_empty_figure("信頼度/dB値データがありません")

        fig = go.Figure(data=[
            go.Histogram(
                x=df[data_col],
                nbinsx=self._get_max_bins(),
                name='分布',
                marker_color=self.colors[1],
                opacity=0.7
            )
        ])

        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title="件数",
            template=self.template,
            showlegend=False
        )

        return fig
    
    def create_response_time_boxplot(self, df: pd.DataFrame) -> go.Figure:
        """応答時間の箱ひげ図"""
        if df.empty:
            return self._create_empty_figure("データがありません")

        # M5Stackには応答時間がないため、dB値の箱ひげ図を表示
        if 'm5stack_db_value' in df.columns and not df['m5stack_db_value'].isna().all():
            data_col = 'm5stack_db_value'
            title = "カテゴリ別M5Stack検出dB値分布"
            yaxis_title = "dB値"
        elif 'dummy_response_time' in df.columns:
            data_col = 'dummy_response_time'
            title = "カテゴリ別ダミー応答時間分布（参考）"
            yaxis_title = "応答時間 (秒)"
        elif 'response_time' in df.columns:
            # 旧形式との互換性
            data_col = 'response_time'
            title = "カテゴリ別応答時間分布"
            yaxis_title = "応答時間 (秒)"
        else:
            return self._create_empty_figure("応答時間/dB値データがありません")

        categories = df['category'].unique()

        fig = go.Figure()

        for i, category in enumerate(categories):
            cat_data = df[df['category'] == category]
            fig.add_trace(go.Box(
                y=cat_data[data_col],
                name=category,
                marker_color=self.colors[i % len(self.colors)]
            ))

        fig.update_layout(
            title=title,
            xaxis_title="音声カテゴリ",
            yaxis_title=yaxis_title,
            template=self.template
        )

        return fig
    
    def create_performance_timeline(self, df: pd.DataFrame) -> go.Figure:
        """性能推移の時系列グラフ"""
        if df.empty:
            return self._create_empty_figure("データがありません")

        # M5Stackデータを優先的に使用
        if 'avg_db' in df.columns and not df['avg_db'].isna().all():
            subplot2_title = "平均dB値の推移"
            data2_col = 'avg_db'
            yaxis2_title = "dB値"
            trace2_name = '平均dB値'
        elif 'confidence' in df.columns:
            subplot2_title = "平均信頼度の推移"
            data2_col = 'confidence'
            yaxis2_title = "信頼度"
            trace2_name = '平均信頼度'
        else:
            # 第2グラフなしで精度のみ表示
            fig = go.Figure(data=[
                go.Scatter(
                    x=df['datetime'],
                    y=df['accuracy'] * 100,
                    mode='lines+markers',
                    name='識別精度 (%)',
                    line=dict(color=self.colors[0], width=2),
                    marker=dict(size=8)
                )
            ])
            fig.update_layout(
                title="性能推移（時系列）- 識別精度",
                xaxis_title="日時",
                yaxis_title="精度 (%)",
                template=self.template
            )
            return fig

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("識別精度の推移", subplot2_title),
            vertical_spacing=0.1
        )

        # 識別精度
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=df['accuracy'] * 100,
                mode='lines+markers',
                name='識別精度 (%)',
                line=dict(color=self.colors[0], width=2),
                marker=dict(size=8)
            ),
            row=1, col=1
        )

        # 信頼度またはdB値
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=df[data2_col],
                mode='lines+markers',
                name=trace2_name,
                line=dict(color=self.colors[1], width=2),
                marker=dict(size=8)
            ),
            row=2, col=1
        )

        fig.update_layout(
            title="性能推移（時系列）",
            template=self.template,
            height=self._get_timeline_height()
        )

        fig.update_xaxes(title_text="日時", row=2, col=1)
        fig.update_yaxes(title_text="精度 (%)", row=1, col=1)
        fig.update_yaxes(title_text=yaxis2_title, row=2, col=1)

        return fig
    
    def create_waveform_plot(self, waveform_data: Dict, title: str = "音声波形") -> go.Figure:
        """波形データのプロット"""
        if not waveform_data:
            return self._create_empty_figure("波形データがありません")
        
        # サブプロットを作成（上下に2つ）
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("出力電圧", "入力電圧"),
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        
        # 標準解像度データ
        if 'standard' in waveform_data:
            std_data = waveform_data['standard']
            
            # 出力電圧（上段）
            if std_data.get('time') and std_data.get('output_voltage'):
                fig.add_trace(go.Scatter(
                    x=std_data['time'],
                    y=std_data['output_voltage'],
                    mode='lines',
                    name='出力電圧',
                    line=dict(color='red', width=1),
                    showlegend=True
                ), row=1, col=1)
            
            # 入力電圧（下段）
            if std_data.get('time') and std_data.get('input_voltage'):
                fig.add_trace(go.Scatter(
                    x=std_data['time'],
                    y=std_data['input_voltage'],
                    mode='lines',
                    name='入力電圧',
                    line=dict(color='blue', width=1),
                    showlegend=True
                ), row=2, col=1)
        
        
        fig.update_layout(
            title=title,
            template=self.template,
            height=800  # 上下に分割するため高さを増加
        )
        
        # X軸とY軸のラベルを設定
        fig.update_xaxes(title_text="時間 (秒)", row=2, col=1)
        fig.update_yaxes(title_text="電圧 (V)", range=[0, 3.3], row=1, col=1)
        fig.update_yaxes(title_text="電圧 (V)", range=[0, 3.3], row=2, col=1)
        
        return fig
    
    def create_confusion_matrix(self, df: pd.DataFrame) -> go.Figure:
        """混同行列のヒートマップ"""
        if df.empty:
            return self._create_empty_figure("データがありません")

        # M5Stack分類データを使用（存在する場合）
        if 'm5stack_detected_class' in df.columns and not df['m5stack_detected_class'].isna().all():
            predicted_col = 'm5stack_detected_class'
            title = "M5Stack分類結果混同行列"
        elif 'dummy_predicted_category' in df.columns:
            predicted_col = 'dummy_predicted_category'
            title = "ダミー識別結果混同行列（参考）"
        elif 'predicted_category' in df.columns:
            # 旧形式との互換性
            predicted_col = 'predicted_category'
            title = "識別結果混同行列"
        else:
            return self._create_empty_figure("予測データがありません")

        # 混同行列を作成
        categories = sorted(df['category'].unique())
        matrix = pd.crosstab(
            df['category'],
            df[predicted_col],
            margins=False
        )

        # 全カテゴリが含まれるようにreindex
        matrix = matrix.reindex(index=categories, columns=categories, fill_value=0)

        fig = go.Figure(data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale='Blues',
            text=matrix.values,
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False
        ))

        fig.update_layout(
            title=title,
            xaxis_title="予測カテゴリ",
            yaxis_title="実際のカテゴリ",
            template=self.template
        )
        
        return fig
    
    def create_test_summary_cards(self, df: pd.DataFrame) -> Dict:
        """テストサマリーのカード情報を生成"""
        if df.empty:
            return {
                'total_tests': 0,
                'accuracy': 0.0,
                'avg_confidence': 0.0,
                'avg_response_time': 0.0,
                'avg_db': 0.0
            }

        # M5Stack分類データを優先的に使用
        if 'm5stack_correct' in df.columns and not df['m5stack_correct'].isna().all():
            accuracy = df['m5stack_correct'].mean() * 100
            avg_db = df['m5stack_db_value'].mean() if 'm5stack_db_value' in df.columns else 0.0
            avg_confidence = 0.0  # M5Stackには信頼度なし
            avg_response_time = 0.0  # M5Stackには応答時間なし
        elif 'dummy_identification_correct' in df.columns:
            accuracy = df['dummy_identification_correct'].mean() * 100
            avg_confidence = df['dummy_confidence'].mean() if 'dummy_confidence' in df.columns else 0.0
            avg_response_time = df['dummy_response_time'].mean() if 'dummy_response_time' in df.columns else 0.0
            avg_db = 0.0
        elif 'identification_correct' in df.columns:
            # 旧形式との互換性
            accuracy = df['identification_correct'].mean() * 100
            avg_confidence = df['confidence'].mean() if 'confidence' in df.columns else 0.0
            avg_response_time = df['response_time'].mean() if 'response_time' in df.columns else 0.0
            avg_db = 0.0
        else:
            accuracy = 0.0
            avg_confidence = 0.0
            avg_response_time = 0.0
            avg_db = 0.0

        return {
            'total_tests': len(df),
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'avg_response_time': avg_response_time,
            'avg_db': avg_db
        }
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """空のグラフを作成"""
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text=message,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            template=self.template,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    def _get_default_height(self) -> int:
        """デフォルトグラフ高さを取得"""
        from ..config import config
        return config.DEFAULT_FIGURE_HEIGHT
    
    def _get_timeline_height(self) -> int:
        """タイムライングラフ高さを取得"""
        from ..config import config
        return config.TIMELINE_FIGURE_HEIGHT
    
    def _get_max_bins(self) -> int:
        """ヒストグラムの最大ビン数を取得"""
        from ..config import config
        return config.MAX_HISTOGRAM_BINS


# グローバルなグラフジェネレーターインスタンス
graph_generator = GraphGenerator()