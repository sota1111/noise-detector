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
        
        # カテゴリ別精度を計算
        category_stats = df.groupby('category').agg({
            'identification_correct': ['mean', 'count']
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
            title="カテゴリ別識別精度",
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
        
        fig = go.Figure(data=[
            go.Histogram(
                x=df['confidence'],
                nbinsx=self._get_max_bins(),
                name='信頼度分布',
                marker_color=self.colors[1],
                opacity=0.7
            )
        ])
        
        fig.update_layout(
            title="識別信頼度分布",
            xaxis_title="信頼度",
            yaxis_title="件数",
            template=self.template,
            showlegend=False
        )
        
        return fig
    
    def create_response_time_boxplot(self, df: pd.DataFrame) -> go.Figure:
        """応答時間の箱ひげ図"""
        if df.empty:
            return self._create_empty_figure("データがありません")
        
        categories = df['category'].unique()
        
        fig = go.Figure()
        
        for i, category in enumerate(categories):
            cat_data = df[df['category'] == category]
            fig.add_trace(go.Box(
                y=cat_data['response_time'],
                name=category,
                marker_color=self.colors[i % len(self.colors)]
            ))
        
        fig.update_layout(
            title="カテゴリ別応答時間分布",
            xaxis_title="音声カテゴリ",
            yaxis_title="応答時間 (秒)",
            template=self.template
        )
        
        return fig
    
    def create_performance_timeline(self, df: pd.DataFrame) -> go.Figure:
        """性能推移の時系列グラフ"""
        if df.empty:
            return self._create_empty_figure("データがありません")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("識別精度の推移", "平均信頼度の推移"),
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
        
        # 信頼度
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=df['confidence'],
                mode='lines+markers',
                name='平均信頼度',
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
        fig.update_yaxes(title_text="信頼度", row=2, col=1)
        
        return fig
    
    def create_waveform_plot(self, waveform_data: Dict, title: str = "音声波形") -> go.Figure:
        """波形データのプロット"""
        if not waveform_data:
            return self._create_empty_figure("波形データがありません")
        
        fig = go.Figure()
        
        # 標準解像度データ
        if 'standard' in waveform_data:
            std_data = waveform_data['standard']
            if std_data.get('time') and std_data.get('input_voltage'):
                fig.add_trace(go.Scatter(
                    x=std_data['time'],
                    y=std_data['input_voltage'],
                    mode='lines',
                    name='入力電圧',
                    line=dict(color=self.colors[0], width=1)
                ))
            
            if std_data.get('time') and std_data.get('output_voltage'):
                fig.add_trace(go.Scatter(
                    x=std_data['time'],
                    y=std_data['output_voltage'],
                    mode='lines',
                    name='出力電圧',
                    line=dict(color=self.colors[1], width=1)
                ))
        
        # 高解像度データ（最初の1秒）
        if 'highres' in waveform_data:
            hr_data = waveform_data['highres']
            if hr_data.get('time') and hr_data.get('input_voltage'):
                fig.add_trace(go.Scatter(
                    x=hr_data['time'],
                    y=hr_data['input_voltage'],
                    mode='lines',
                    name='入力電圧（高解像度）',
                    line=dict(color=self.colors[2], width=2),
                    visible='legendonly'  # デフォルトで非表示
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="時間 (秒)",
            yaxis_title="電圧 (V)",
            template=self.template,
            height=self._get_default_height()
        )
        
        return fig
    
    def create_confusion_matrix(self, df: pd.DataFrame) -> go.Figure:
        """混同行列のヒートマップ"""
        if df.empty:
            return self._create_empty_figure("データがありません")
        
        # 混同行列を作成
        categories = sorted(df['category'].unique())
        matrix = pd.crosstab(
            df['category'], 
            df['predicted_category'],
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
            title="識別結果混同行列",
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
                'avg_response_time': 0.0
            }
        
        return {
            'total_tests': len(df),
            'accuracy': df['identification_correct'].mean() * 100,
            'avg_confidence': df['confidence'].mean(),
            'avg_response_time': df['response_time'].mean()
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