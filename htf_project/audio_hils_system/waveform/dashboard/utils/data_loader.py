"""
データ読み込みユーティリティ
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import glob


class DataLoader:
    """テスト結果データの読み込み・管理クラス"""
    
    def __init__(self, base_path: str = None):
        """
        Args:
            base_path: プロジェクトのベースパス
        """
        if base_path is None:
            # dashboard/utils/data_loader.pyから相対的にプロジェクトルートを取得
            current_file = Path(__file__).resolve()
            self.base_path = current_file.parents[4]  # htf_project/
        else:
            self.base_path = Path(base_path)
        
        self.data_output_path = self.base_path / "audio_hils_system" / "waveform" / "data" / "output"
        self.output_path = self.base_path / "output"
        
        # 両方のレポートパスを検索対象に
        self.reports_paths = [
            self.output_path / "reports",  # 新しい場所（最新セッション）
            self.data_output_path / "reports"  # 古い場所（既存セッション）
        ]
        self.csv_path = self.output_path / "csv"
        self.graphs_path = self.output_path / "graphs"
    
    def get_available_sessions(self) -> List[Dict]:
        """利用可能なテストセッション一覧を取得（複数パス対応）"""
        sessions = []
        
        # 複数のレポートパスを検索
        for reports_path in self.reports_paths:
            if not reports_path.exists():
                continue
                
            # セッションサマリーファイルから情報を取得
            summary_files = glob.glob(str(reports_path / "session_*_summary.md"))
            
            for summary_file in summary_files:
                try:
                    session_id = Path(summary_file).name.replace("session_", "").replace("_summary.md", "")
                    
                    # 既に追加済みのセッションはスキップ
                    if any(s['session_id'] == session_id for s in sessions):
                        continue
                    
                    # 対応するCSVファイルを探す（複数パターン対応）
                    csv_patterns = [
                        reports_path / f"{session_id}_statistics.csv",
                        reports_path / f"*{session_id}_statistics.csv"
                    ]
                    
                    csv_file = None
                    for pattern in csv_patterns:
                        csv_matches = glob.glob(str(pattern))
                        if csv_matches:
                            csv_file = csv_matches[0]  # 最初のマッチを使用
                            break
                    
                    # バッチレポートファイルを探す
                    batch_reports = glob.glob(str(reports_path / f"*{session_id}_batch_report.md"))
                    
                    sessions.append({
                        'session_id': session_id,
                        'summary_file': summary_file,
                        'csv_file': csv_file,
                        'batch_reports': batch_reports,
                        'reports_path': str(reports_path)  # どのパスから取得したかを記録
                    })
                except Exception as e:
                    print(f"セッション情報の取得エラー ({summary_file}): {e}")
                    continue
        
        # セッションIDで降順ソート（最新が先頭）
        return sorted(sessions, key=lambda x: x['session_id'], reverse=True)
    
    def load_session_statistics(self, session_id: str) -> Optional[pd.DataFrame]:
        """特定セッションの統計データを読み込み（複数パス対応）"""
        try:
            # 複数のレポートパスでCSVファイルを探す
            csv_files = []
            for reports_path in self.reports_paths:
                csv_pattern = str(reports_path / f"*{session_id}_statistics.csv")
                csv_files.extend(glob.glob(csv_pattern))
            
            if not csv_files:
                print(f"セッション {session_id} の統計CSVファイルが見つかりません")
                return None
            
            # 最新のCSVファイルを読み込み（タイムスタンプ順）
            csv_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            df = pd.read_csv(csv_files[0])
            print(f"統計データ読み込み成功: {csv_files[0]} ({len(df)} 行)")
            return df
            
        except Exception as e:
            print(f"統計データの読み込みエラー ({session_id}): {e}")
            return None
    
    def load_all_statistics(self) -> pd.DataFrame:
        """全セッションの統計データを統合して読み込み"""
        all_data = []
        sessions = self.get_available_sessions()
        
        for session in sessions:
            if session['csv_file'] and os.path.exists(session['csv_file']):
                try:
                    df = pd.read_csv(session['csv_file'])
                    df['session_id'] = session['session_id']
                    all_data.append(df)
                except Exception as e:
                    print(f"統計データの読み込みエラー ({session['session_id']}): {e}")
                    continue
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_session_summary(self, session_id: str) -> Dict:
        """セッションサマリー情報を取得"""
        sessions = self.get_available_sessions()
        session_data = next((s for s in sessions if s['session_id'] == session_id), None)
        
        if not session_data or not os.path.exists(session_data['summary_file']):
            return {}
        
        try:
            # Markdownファイルからサマリー情報を解析
            with open(session_data['summary_file'], 'r', encoding='utf-8') as f:
                content = f.read()
            
            summary = {
                'session_id': session_id,
                'raw_content': content
            }
            
            # 簡単な情報抽出
            lines = content.split('\n')
            for line in lines:
                try:
                    if '総テスト数:' in line:
                        value = line.split(':')[1].strip()
                        # 数値以外の文字を除去
                        numeric_value = ''.join(filter(str.isdigit, value))
                        if numeric_value:
                            summary['total_tests'] = int(numeric_value)
                    elif '成功テスト数:' in line:
                        value = line.split(':')[1].strip()
                        numeric_value = ''.join(filter(str.isdigit, value))
                        if numeric_value:
                            summary['success_tests'] = int(numeric_value)
                    elif '成功率:' in line:
                        value = line.split(':')[1].strip().replace('%', '')
                        # 小数点と数字のみを抽出
                        import re
                        numeric_match = re.search(r'[\d.]+', value)
                        if numeric_match:
                            summary['success_rate'] = float(numeric_match.group())
                except (ValueError, IndexError) as e:
                    print(f"サマリー行の解析エラー ({line}): {e}")
                    continue
            
            return summary
            
        except Exception as e:
            print(f"サマリー情報の取得エラー: {e}")
            return {'session_id': session_id, 'error': str(e)}
    
    def get_available_categories(self) -> List[str]:
        """利用可能な音声カテゴリ一覧を取得"""
        df = self.load_all_statistics()
        if df.empty:
            return []
        
        categories = df['category'].unique().tolist()
        return sorted([cat for cat in categories if pd.notna(cat)])
    
    def get_waveform_data(self, filename: str) -> Optional[Dict]:
        """波形データを取得"""
        try:
            # CSVファイルから波形データを読み込み
            csv_files = glob.glob(str(self.csv_path / f"*{filename.split('.')[0]}*.csv"))
            
            waveform_data = {}
            for csv_file in csv_files:
                if 'first1s_highres' in csv_file:
                    # 高解像度データ
                    df = pd.read_csv(csv_file)
                    # 実際の列名に対応
                    time_col = 'Time(s)' if 'Time(s)' in df.columns else 'time'
                    input_col = 'Input_Voltage(V)' if 'Input_Voltage(V)' in df.columns else 'input_voltage'
                    output_col = 'Output_Voltage(V)' if 'Output_Voltage(V)' in df.columns else 'output_voltage'
                    
                    waveform_data['highres'] = {
                        'time': df[time_col].tolist() if time_col in df.columns else [],
                        'input_voltage': df[input_col].tolist() if input_col in df.columns else [],
                        'output_voltage': df[output_col].tolist() if output_col in df.columns else []
                    }
                else:
                    # 標準データ
                    df = pd.read_csv(csv_file)
                    # 実際の列名に対応
                    time_col = 'Time(s)' if 'Time(s)' in df.columns else 'time'
                    input_col = 'Input_Voltage(V)' if 'Input_Voltage(V)' in df.columns else 'input_voltage'
                    output_col = 'Output_Voltage(V)' if 'Output_Voltage(V)' in df.columns else 'output_voltage'
                    
                    waveform_data['standard'] = {
                        'time': df[time_col].tolist() if time_col in df.columns else [],
                        'input_voltage': df[input_col].tolist() if input_col in df.columns else [],
                        'output_voltage': df[output_col].tolist() if output_col in df.columns else []
                    }
            
            return waveform_data if waveform_data else None
            
        except Exception as e:
            print(f"波形データの取得エラー: {e}")
            return None
    
    def get_latest_session_id(self) -> Optional[str]:
        """最新のセッションIDを取得"""
        sessions = self.get_available_sessions()
        return sessions[0]['session_id'] if sessions else None
    
    def get_test_performance_over_time(self) -> pd.DataFrame:
        """時系列での性能推移データを取得"""
        df = self.load_all_statistics()
        if df.empty:
            return pd.DataFrame()
        
        # セッションIDから日時情報を抽出
        df['datetime'] = pd.to_datetime(df['session_id'], format='%Y%m%d_%H%M%S', errors='coerce')
        
        # セッション別の集計
        session_stats = df.groupby(['session_id', 'datetime']).agg({
            'identification_correct': 'mean',
            'confidence': 'mean',
            'response_time': 'mean',
            'test_number': 'count'
        }).reset_index()
        
        session_stats.rename(columns={
            'identification_correct': 'accuracy',
            'test_number': 'test_count'
        }, inplace=True)
        
        return session_stats.sort_values('datetime')


# グローバルなデータローダーインスタンス
data_loader = DataLoader()