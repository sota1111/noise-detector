"""
テストレポート生成モジュール
音声識別評価システムのテスト結果からレポートを生成
"""

import os
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from ..config.config import OUTPUT_CSV_FOLDER, OUTPUT_GRAPH_FOLDER


class ReportGenerator:
    """テストレポート生成クラス"""
    
    def __init__(self, report_folder="output/reports"):
        """
        レポート生成器を初期化
        
        Args:
            report_folder (str): レポート出力フォルダ
        """
        self.report_folder = report_folder
        self.csv_folder = OUTPUT_CSV_FOLDER
        self.graph_folder = OUTPUT_GRAPH_FOLDER
        
        # レポートフォルダ作成
        os.makedirs(self.report_folder, exist_ok=True)
    
    def generate_single_test_report(self, test_result):
        """
        単一テスト結果のレポートを生成
        
        Args:
            test_result (dict): テスト結果
            
        Returns:
            str: レポートファイルパス
        """
        if not test_result.get('success', False):
            return self._generate_error_report(test_result)
        
        # レポートファイル名
        labeled_base = test_result['labeled_filename_base']
        report_filename = f"{labeled_base}_report.md"
        report_path = os.path.join(self.report_folder, report_filename)
        
        # レポート内容生成
        report_content = self._create_single_test_report_content(test_result)
        
        # ファイル書き込み
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✓ 単一テストレポート生成: {report_path}")
        return report_path
    
    def generate_batch_test_report(self, batch_result):
        """
        バッチテスト結果のレポートを生成
        
        Args:
            batch_result (dict): バッチテスト結果
            
        Returns:
            str: レポートファイルパス
        """
        # レポートファイル名
        test_name = batch_result.get('test_name', 'batch_test')
        session_id = batch_result.get('session_id', 'unknown')
        report_filename = f"{test_name}_batch_report.md"
        report_path = os.path.join(self.report_folder, report_filename)
        
        # レポート内容生成
        report_content = self._create_batch_test_report_content(batch_result)
        
        # ファイル書き込み
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # 統計CSVも生成
        csv_path = self._generate_batch_statistics_csv(batch_result)
        
        print(f"✓ バッチテストレポート生成: {report_path}")
        if csv_path:
            print(f"✓ バッチテスト統計CSV生成: {csv_path}")
        
        return report_path
    
    def _create_single_test_report_content(self, test_result):
        """単一テストレポートの内容を作成"""
        content = []
        
        # ヘッダー
        content.append(f"# 音声識別評価テストレポート")
        content.append(f"")
        content.append(f"**生成日時:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"**テストセッション:** {test_result.get('session_id', 'N/A')}")
        content.append(f"")
        
        # テスト概要
        content.append(f"## テスト概要")
        content.append(f"")
        content.append(f"- **テスト名:** {test_result.get('test_name', 'N/A')}")
        content.append(f"- **音声ファイル:** {Path(test_result['file_path']).name}")
        content.append(f"- **ファイルパス:** `{test_result['file_path']}`")
        content.append(f"- **カテゴリ:** {test_result.get('category', 'N/A')}")
        content.append(f"- **テスト時間:** {test_result.get('test_duration', 0)}秒")
        content.append(f"- **実行時間:** {test_result.get('execution_time', 0):.2f}秒")
        content.append(f"")
        
        # ファイル情報
        file_info = test_result.get('file_info', {})
        content.append(f"## 音声ファイル情報")
        content.append(f"")
        content.append(f"- **サンプリングレート:** {file_info.get('sample_rate', 'N/A')} Hz")
        content.append(f"- **ファイル長:** {file_info.get('file_duration', 0):.2f}秒")
        content.append(f"- **総サンプル数:** {file_info.get('total_samples', 'N/A')}")
        content.append(f"- **チャンク数:** {file_info.get('num_chunks', 'N/A')}")
        content.append(f"")
        
        # データ処理統計
        data_summary = test_result.get('data_summary', {})
        content.append(f"## データ処理統計")
        content.append(f"")
        content.append(f"- **処理チャンク数:** {data_summary.get('chunks_processed', 'N/A')}")
        content.append(f"- **総処理サンプル数:** {data_summary.get('total_samples', 'N/A')}")
        content.append(f"- **サンプリングレート:** {data_summary.get('sample_rate', 'N/A')} Hz")
        content.append(f"")
        
        # 音声識別結果
        identification_result = test_result.get('identification_result', {})
        if identification_result:
            content.append(f"## 音声識別結果")
            content.append(f"")
            content.append(f"- **予測カテゴリ:** {identification_result.get('predicted_category', 'N/A')}")
            content.append(f"- **正解カテゴリ:** {identification_result.get('true_category', 'N/A')}")
            content.append(f"- **識別結果:** {'✓ 正解' if identification_result.get('is_correct', False) else '× 不正解'}")
            content.append(f"- **信頼度:** {identification_result.get('confidence', 0):.3f}")
            content.append(f"- **応答時間:** {identification_result.get('response_time', 0):.3f}秒")
            
            # 上位候補
            top_candidates = identification_result.get('top_candidates', [])
            if len(top_candidates) > 1:
                content.append(f"- **上位候補:** {', '.join(top_candidates[:3])}")
            content.append(f"")
        
        # 出力ファイル
        output_files = test_result.get('output_files', {})
        content.append(f"## 出力ファイル")
        content.append(f"")
        content.append(f"### データファイル")
        content.append(f"- **CSV:** `{output_files.get('csv', 'N/A')}`")
        content.append(f"- **高解像度CSV:** `{output_files.get('highres_csv', 'N/A')}`")
        content.append(f"")
        content.append(f"### グラフファイル")
        content.append(f"- **連続データグラフ:** `{output_files.get('graph', 'N/A')}`")
        content.append(f"- **高解像度グラフ:** `{output_files.get('highres_graph', 'N/A')}`")
        content.append(f"")
        
        # テスト設定
        content.append(f"## テスト設定")
        content.append(f"")
        content.append(f"- **AD3出力範囲:** ±1.65V (3.3V p-p)")
        content.append(f"- **AD3スコープ範囲:** ±5V")
        content.append(f"- **チャンク時間:** 1.0秒")
        content.append(f"- **正規化:** 5秒間全体基準")
        content.append(f"")
        
        return '\n'.join(content)
    
    def _create_batch_test_report_content(self, batch_result):
        """バッチテストレポートの内容を作成"""
        content = []
        
        # ヘッダー
        content.append(f"# バッチ音声識別評価テストレポート")
        content.append(f"")
        content.append(f"**生成日時:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"**テストセッション:** {batch_result.get('session_id', 'N/A')}")
        content.append(f"")
        
        # テスト概要
        content.append(f"## テスト概要")
        content.append(f"")
        content.append(f"- **テスト名:** {batch_result.get('test_name', 'N/A')}")
        content.append(f"- **テストタイプ:** {batch_result.get('test_type', 'batch')}")
        content.append(f"- **対象カテゴリ:** {batch_result.get('category', 'N/A')}")
        content.append(f"- **総ファイル数:** {batch_result.get('total_files', 0)}")
        content.append(f"- **成功ファイル数:** {batch_result.get('successful_files', 0)}")
        content.append(f"- **失敗ファイル数:** {batch_result.get('failed_files', 0)}")
        content.append(f"- **成功率:** {(batch_result.get('successful_files', 0) / max(batch_result.get('total_files', 1), 1) * 100):.1f}%")
        content.append(f"- **総実行時間:** {batch_result.get('execution_time', 0):.2f}秒")
        content.append(f"")
        
        # 統計情報
        statistics = batch_result.get('statistics', {})
        content.append(f"## テスト統計")
        content.append(f"")
        content.append(f"- **総テスト時間:** {statistics.get('total_test_duration', 0)}秒")
        content.append(f"- **平均実行時間/ファイル:** {statistics.get('average_execution_time', 0):.2f}秒")
        content.append(f"- **総処理サンプル数:** {statistics.get('total_samples_processed', 'N/A')}")
        content.append(f"")
        
        # 音声識別統計
        identification_stats = batch_result.get('identification_statistics')
        if identification_stats:
            content.append(f"## 音声識別統計")
            content.append(f"")
            content.append(f"- **識別正解率:** {identification_stats.get('accuracy', 0)*100:.1f}%")
            content.append(f"- **正解数/総数:** {identification_stats.get('correct_count', 0)}/{identification_stats.get('total_count', 0)}")
            content.append(f"- **平均信頼度:** {identification_stats.get('average_confidence', 0):.3f}")
            content.append(f"- **平均応答時間:** {identification_stats.get('average_response_time', 0):.3f}秒")
            
            # カテゴリ別正解率
            category_accuracy = identification_stats.get('category_accuracy', {})
            if category_accuracy:
                content.append(f"")
                content.append(f"### カテゴリ別正解率")
                content.append(f"")
                content.append(f"| カテゴリ | 正解率 | 正解数/総数 |")
                content.append(f"|----------|--------|-------------|")
                
                for category, accuracy in sorted(category_accuracy.items()):
                    category_stats = identification_stats.get('category_stats', {}).get(category, {})
                    correct = category_stats.get('correct', 0)
                    total = category_stats.get('total', 0)
                    content.append(f"| {category} | {accuracy*100:.1f}% | {correct}/{total} |")
            
            content.append(f"")
        
        # 個別テスト結果
        content.append(f"## 個別テスト結果")
        content.append(f"")
        content.append(f"| No. | ファイル名 | カテゴリ | テスト結果 | 識別結果 | 予測 | 信頼度 | 実行時間 |")
        content.append(f"|-----|-----------|----------|------------|----------|------|--------|----------|")
        
        results = batch_result.get('results', [])
        for idx, result in enumerate(results, 1):
            filename = Path(result['file_path']).name if 'file_path' in result else 'N/A'
            category = result.get('category', 'N/A')
            test_status = "✓ 成功" if result.get('success', False) else "× 失敗"
            exec_time = f"{result.get('execution_time', 0):.2f}s"
            
            # 識別結果情報
            identification_result = result.get('identification_result', {})
            if identification_result:
                identification_status = "✓ 正解" if identification_result.get('is_correct', False) else "× 不正解"
                predicted = identification_result.get('predicted_category', 'N/A')
                confidence = f"{identification_result.get('confidence', 0):.3f}"
            else:
                identification_status = "N/A"
                predicted = "N/A"
                confidence = "N/A"
            
            content.append(f"| {idx} | {filename} | {category} | {test_status} | {identification_status} | {predicted} | {confidence} | {exec_time} |")
        
        content.append(f"")
        
        # 失敗詳細（失敗があった場合）
        failed_results = [r for r in results if not r.get('success', False)]
        if failed_results:
            content.append(f"## 失敗詳細")
            content.append(f"")
            for idx, result in enumerate(failed_results, 1):
                filename = Path(result['file_path']).name if 'file_path' in result else 'N/A'
                error = result.get('error', '不明なエラー')
                content.append(f"### {idx}. {filename}")
                content.append(f"**エラー:** {error}")
                content.append(f"")
        
        return '\n'.join(content)
    
    def _generate_batch_statistics_csv(self, batch_result):
        """バッチテスト統計CSVを生成"""
        try:
            results = batch_result.get('results', [])
            if not results:
                return None
            
            # データフレーム作成
            data = []
            for idx, result in enumerate(results, 1):
                # 識別結果情報
                identification_result = result.get('identification_result', {})
                
                row = {
                    'test_number': idx,
                    'filename': Path(result['file_path']).name if 'file_path' in result else 'N/A',
                    'file_path': result.get('file_path', ''),
                    'category': result.get('category', ''),
                    'success': result.get('success', False),
                    'test_duration': result.get('test_duration', 0),
                    'execution_time': result.get('execution_time', 0),
                    'error': result.get('error', '') if not result.get('success', False) else '',
                    'chunks_processed': result.get('data_summary', {}).get('chunks_processed', 0),
                    'total_samples': result.get('data_summary', {}).get('total_samples', 0),
                    'predicted_category': identification_result.get('predicted_category', ''),
                    'identification_correct': identification_result.get('is_correct', False),
                    'confidence': identification_result.get('confidence', 0),
                    'response_time': identification_result.get('response_time', 0)
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # CSV保存
            test_name = batch_result.get('test_name', 'batch_test')
            csv_filename = f"{test_name}_statistics.csv"
            csv_path = os.path.join(self.report_folder, csv_filename)
            
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            return csv_path
            
        except Exception as e:
            print(f"統計CSV生成エラー: {e}")
            return None
    
    def _generate_error_report(self, test_result):
        """エラーレポートを生成"""
        # エラーレポートファイル名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"error_report_{timestamp}.md"
        report_path = os.path.join(self.report_folder, report_filename)
        
        content = []
        content.append(f"# エラーレポート")
        content.append(f"")
        content.append(f"**生成日時:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"")
        content.append(f"## エラー詳細")
        content.append(f"")
        content.append(f"- **ファイル:** {test_result.get('file_path', 'N/A')}")
        content.append(f"- **エラー:** {test_result.get('error', '不明なエラー')}")
        content.append(f"- **実行時間:** {test_result.get('execution_time', 0):.2f}秒")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        print(f"✓ エラーレポート生成: {report_path}")
        return report_path
    
    def generate_session_summary(self, session_id, test_results):
        """
        セッション全体のサマリーレポートを生成
        
        Args:
            session_id (str): セッションID
            test_results (list): テスト結果のリスト
            
        Returns:
            str: サマリーレポートファイルパス
        """
        # サマリーファイル名
        summary_filename = f"session_{session_id}_summary.md"
        summary_path = os.path.join(self.report_folder, summary_filename)
        
        # 統計計算
        total_tests = len(test_results)
        successful_tests = len([r for r in test_results if r.get('success', False)])
        failed_tests = total_tests - successful_tests
        
        # カテゴリ別統計
        category_stats = {}
        for result in test_results:
            if result.get('success', False):
                category = result.get('category', 'Unknown')
                if category not in category_stats:
                    category_stats[category] = {'success': 0, 'total': 0}
                category_stats[category]['success'] += 1
            
            category = result.get('category', 'Unknown')
            if category not in category_stats:
                category_stats[category] = {'success': 0, 'total': 0}
            category_stats[category]['total'] += 1
        
        # レポート内容生成
        content = []
        content.append(f"# セッションサマリーレポート")
        content.append(f"")
        content.append(f"**セッションID:** {session_id}")
        content.append(f"**生成日時:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"")
        
        content.append(f"## 全体統計")
        content.append(f"")
        content.append(f"- **総テスト数:** {total_tests}")
        content.append(f"- **成功テスト数:** {successful_tests}")
        content.append(f"- **失敗テスト数:** {failed_tests}")
        content.append(f"- **成功率:** {(successful_tests / max(total_tests, 1) * 100):.1f}%")
        content.append(f"")
        
        if category_stats:
            content.append(f"## カテゴリ別統計")
            content.append(f"")
            content.append(f"| カテゴリ | 成功 | 合計 | 成功率 |")
            content.append(f"|----------|------|------|--------|")
            
            for category, stats in sorted(category_stats.items()):
                success_rate = (stats['success'] / max(stats['total'], 1) * 100)
                content.append(f"| {category} | {stats['success']} | {stats['total']} | {success_rate:.1f}% |")
            
            content.append(f"")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        print(f"✓ セッションサマリー生成: {summary_path}")
        return summary_path