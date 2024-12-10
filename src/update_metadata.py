import os
import pandas as pd
import librosa
from musicnn.extractor import extractor
from musicnn.tagger import top_tags

# 경로 설정
AUDIO_DIR = "/content/drive/MyDrive/프로젝트/케어크루즈 인턴/data/maestro_v3/processed_musicgen"
METADATA_FILE = "/content/drive/MyDrive/프로젝트/케어크루즈 인턴/data/maestro_v3/processed_musicgen/processed_metadata.csv"
OUTPUT_METADATA_FILE = "/content/drive/MyDrive/프로젝트/케어크루즈 인턴/data/maestro_v3/processed_musicgen/extended_metadata.csv"

# Musicnn을 사용한 태깅
def extract_tags(audio_file):
    """
    음악 파일에서 태그를 예측합니다.
    """
    try:
        # top_tags 함수로 상위 태그 추출
        tags = top_tags(audio_file)
        print(f"Extracted top tags: {tags}")

        # extractor로 추가 정보 추출
        features, tags_list, additional_data = extractor(audio_file)
        print(f"Extracted tags list: {tags_list}")
        return tags, tags_list  # 상위 태그와 태그 리스트 반환
    except Exception as e:
        print(f"Error extracting tags for {audio_file}: {e}")
        return [], []

# RMS/BPM 계산
def calculate_audio_features(audio_file):
    """
    오디오 파일에서 RMS(음량)와 BPM(템포)을 계산합니다.
    """
    try:
        y, sr = librosa.load(audio_file, sr=None)
        rms = librosa.feature.rms(y=y).mean()
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return rms, tempo
    except Exception as e:
        print(f"Error calculating audio features for {audio_file}: {e}")
        return 0.0, 0.0

# 메타데이터 업데이트 함수
def update_metadata_with_features_and_tags(audio_dir, metadata_file, output_metadata_file):
    """
    기존 메타데이터에 태그 및 추가 특성을 포함한 정보를 추가합니다.
    """
    # 메타데이터 읽기
    metadata = pd.read_csv(metadata_file)
    metadata["Top_Tags"] = ""
    metadata["Tags_List"] = ""
    metadata["RMS"] = 0.0
    metadata["Tempo"] = 0.0

    for index, row in metadata.iterrows():
        audio_file = os.path.join(audio_dir, row["audio"])
        if not os.path.exists(audio_file):
            print(f"파일이 존재하지 않습니다: {audio_file}")
            continue

        try:
            # 태그 추출
            top_tags, tags_list = extract_tags(audio_file)

            # RMS/BPM 계산
            rms, tempo = calculate_audio_features(audio_file)

            # 메타데이터 업데이트
            metadata.at[index, "Top_Tags"] = ", ".join(top_tags)
            metadata.at[index, "Tags_List"] = ", ".join(tags_list)
            metadata.at[index, "RMS"] = rms
            metadata.at[index, "Tempo"] = tempo

            print(f"{audio_file}: 태깅 완료 - Top Tags: {top_tags}, RMS: {rms}, Tempo: {tempo}")
        except Exception as e:
            print(f"처리 실패: {audio_file}, 오류: {e}")

    # 확장된 메타데이터 저장
    metadata.to_csv(output_metadata_file, index=False)
    print(f"확장된 메타데이터가 저장되었습니다: {output_metadata_file}")

# 실행
update_metadata_with_features_and_tags(AUDIO_DIR, METADATA_FILE, OUTPUT_METADATA_FILE)
