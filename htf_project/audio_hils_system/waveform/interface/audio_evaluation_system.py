"""
音声識別評価システム メインプログラム
既存のaudio_hils_systemコンポーネントを統合
"""

import sys
import argparse
from pathlib import Path

from ..testing.test_engine import TestEngine
from ..dataset.audio_selector import AudioSelector
from ..testing.report_generator import ReportGenerator


class AudioEvaluationSystem:
    """音声識別評価システム メインクラス"""
    
    def __init__(self):
        """システムを初期化"""
        self.test_engine = TestEngine()
        self.audio_selector = AudioSelector()
        self.report_generator = ReportGenerator()
        self.session_results = []
    
    def initialize(self):
        """システムを初期化"""
        print("=== 音声識別評価システム初期化 ===")
        
        if not self.test_engine.initialize():
            print("× システム初期化に失敗しました")
            return False
        
        print("✓ システム初期化完了")
        return True
    
    def shutdown(self):
        """システムを終了"""
        self.test_engine.shutdown()
        
        # セッションサマリー生成
        if self.session_results and self.test_engine.test_session_id:
            print("\n=== セッションサマリー生成 ===")
            summary_path = self.report_generator.generate_session_summary(
                self.test_engine.test_session_id, 
                self.session_results
            )
        
        print("✓ システム終了")
    
    def run_single_file_test(self, file_path, duration=5, generate_report=True):
        """
        単一ファイルテストを実行
        
        Args:
            file_path (str): 音声ファイルパス
            duration (int): テスト時間（秒）
            generate_report (bool): レポート生成フラグ
            
        Returns:
            dict: テスト結果
        """
        print(f"\n=== 単一ファイルテスト実行 ===")
        print(f"ファイル: {file_path}")
        
        # テスト実行
        result = self.test_engine.run_single_file_test(file_path, duration)
        self.session_results.append(result)
        
        # レポート生成
        if generate_report and result.get('success', False):
            self.report_generator.generate_single_test_report(result)
        
        return result
    
    def run_category_test(self, category, max_files=None, duration_per_file=5, generate_report=True):
        """
        カテゴリテストを実行
        
        Args:
            category (str): ESC-50カテゴリ名
            max_files (int, optional): 最大ファイル数
            duration_per_file (int): ファイルあたりのテスト時間（秒）
            generate_report (bool): レポート生成フラグ
            
        Returns:
            dict: テスト結果
        """
        print(f"\n=== カテゴリテスト実行: {category} ===")
        
        # テスト実行
        result = self.test_engine.run_category_test(category, max_files, duration_per_file)
        
        # 個別結果をセッション結果に追加
        if result.get('success', False):
            self.session_results.extend(result.get('results', []))
        
        # レポート生成
        if generate_report:
            self.report_generator.generate_batch_test_report(result)
        
        return result
    
    def run_random_test(self, file_count=5, duration_per_file=5, generate_report=True):
        """
        ランダムファイルテストを実行
        
        Args:
            file_count (int): テストファイル数
            duration_per_file (int): ファイルあたりのテスト時間（秒）
            generate_report (bool): レポート生成フラグ
            
        Returns:
            dict: テスト結果
        """
        print(f"\n=== ランダムファイルテスト実行 ===")
        print(f"ファイル数: {file_count}")
        
        # テスト実行
        result = self.test_engine.run_random_test(file_count, duration_per_file)
        
        # 個別結果をセッション結果に追加
        if result.get('success', False):
            self.session_results.extend(result.get('results', []))
        
        # レポート生成
        if generate_report:
            self.report_generator.generate_batch_test_report(result)
        
        return result
    
    def run_multiple_files_test(self, file_paths, duration_per_file=5, generate_report=True):
        """
        複数ファイルテストを実行
        
        Args:
            file_paths (list): ファイルパスのリスト
            duration_per_file (int): ファイルあたりのテスト時間（秒）
            generate_report (bool): レポート生成フラグ
            
        Returns:
            dict: テスト結果
        """
        print(f"\n=== 複数ファイルテスト実行 ===")
        print(f"ファイル数: {len(file_paths)}")
        
        # テスト実行
        result = self.test_engine.run_multiple_files_test(file_paths, duration_per_file)
        
        # 個別結果をセッション結果に追加
        if result.get('success', False):
            self.session_results.extend(result.get('results', []))
        
        # レポート生成
        if generate_report:
            self.report_generator.generate_batch_test_report(result)
        
        return result
    
    def list_available_files(self):
        """利用可能なファイル一覧を表示"""
        print("\n=== 利用可能音声ファイル ===")
        self.audio_selector.print_summary()
    
    def list_available_categories(self):
        """利用可能なカテゴリ一覧を表示"""
        categories = self.audio_selector.get_available_categories()
        print(f"\n=== 利用可能カテゴリ ({len(categories)}件) ===")
        for category in categories:
            files = self.audio_selector.get_files_by_category(category)
            print(f"  {category}: {len(files)}ファイル")


def create_argument_parser():
    """コマンドライン引数パーサーを作成"""
    parser = argparse.ArgumentParser(
        description='音声識別評価システム',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 単一ファイルテスト
  python audio_evaluation_system.py --file input/sample.wav --duration 5

  # カテゴリテスト
  python audio_evaluation_system.py --category dog --max-files 3 --duration 3

  # ランダムテスト
  python audio_evaluation_system.py --random 5 --duration 3

  # 利用可能ファイル表示
  python audio_evaluation_system.py --list-files

  # 利用可能カテゴリ表示
  python audio_evaluation_system.py --list-categories
        """
    )
    
    # テストモード選択
    test_group = parser.add_mutually_exclusive_group(required=True)
    test_group.add_argument('--file', type=str, 
                           help='単一音声ファイルのテスト')
    test_group.add_argument('--category', type=str,
                           help='カテゴリ別テスト')
    test_group.add_argument('--random', type=int, metavar='COUNT',
                           help='ランダムファイルテスト（ファイル数指定）')
    test_group.add_argument('--list-files', action='store_true',
                           help='利用可能ファイル一覧表示')
    test_group.add_argument('--list-categories', action='store_true',
                           help='利用可能カテゴリ一覧表示')
    
    # テストパラメータ
    parser.add_argument('--duration', type=int, default=5,
                       help='テスト時間（秒）- デフォルト: 5秒')
    parser.add_argument('--max-files', type=int,
                       help='カテゴリテスト時の最大ファイル数')
    parser.add_argument('--no-report', action='store_true',
                       help='レポート生成を無効化')
    
    return parser


def main():
    """メイン関数"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # システム初期化
    system = AudioEvaluationSystem()
    
    try:
        # リスト系コマンドの場合は接続不要
        if args.list_files:
            system.list_available_files()
            return 0
        
        if args.list_categories:
            system.list_available_categories()
            return 0
        
        # システム初期化（AD3接続含む）
        if not system.initialize():
            return 1
        
        generate_report = not args.no_report
        
        # テスト実行
        if args.file:
            # 単一ファイルテスト
            if not Path(args.file).exists():
                print(f"エラー: ファイルが存在しません: {args.file}")
                return 1
            
            result = system.run_single_file_test(
                args.file, args.duration, generate_report
            )
            
            if not result.get('success', False):
                print(f"テスト失敗: {result.get('error', '不明なエラー')}")
                return 1
        
        elif args.category:
            # カテゴリテスト
            result = system.run_category_test(
                args.category, args.max_files, args.duration, generate_report
            )
            
            if not result.get('success', False):
                print(f"カテゴリテスト失敗: {result.get('error', '不明なエラー')}")
                return 1
        
        elif args.random:
            # ランダムテスト
            result = system.run_random_test(
                args.random, args.duration, generate_report
            )
            
            if not result.get('success', False):
                print(f"ランダムテスト失敗: {result.get('error', '不明なエラー')}")
                return 1
        
        print("\n✓ 全テスト完了")
        return 0
    
    except KeyboardInterrupt:
        print("\n× ユーザーによる中断")
        return 1
    
    except Exception as e:
        print(f"\n× 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        system.shutdown()


if __name__ == "__main__":
    sys.exit(main())