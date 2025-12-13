       # ============ df와 signal 통합 =============
        logger.info("\n🔗 DataFrame과 ML 신호 병합 중...")

        # 신호가 있는 티커만 처리할 새로운 df_dict 생성
        updated_df_dict = {}

        for ticker in ticker_codes:
            #임시
            # df_dict에 없는 티커는 건너뛰기 (데이터가 없는 경우)
            if ticker not in df_dict:
                logger.warning(f"{ticker}: df_dict에 없어 건너뜁니다")
                continue

            df = df_dict[ticker]
            signal = signals.get(ticker)

            #임시
            # 신호가 없는 티커는 건너뛰기 (백테스트에서 제외)
            if signal is None:
                logger.warning(f"{ticker}: 신호가 없어 백테스트에서 제외합니다")
                continue

            # signal은 DatetimeIndex를 가진 Series
            # df는 'date' 컬럼을 가진 DataFrame

            # signal을 DataFrame으로 변환
            signal_df = signal.reset_index()

            # date 타입 맞추기
            df['date'] = pd.to_datetime(df['date'])
            signal_df['date'] = pd.to_datetime(signal_df['date'])

            # 병합 (left join - df의 모든 날짜 유지)
            df = pd.merge(df, signal_df, on='date', how='left')

            # NaN이 있으면 0으로 채우기 (신호가 없는 날은 매수하지 않음)
            df['signal'] = df['signal'].fillna(0)

            # 업데이트된 df_dict에 저장
            updated_df_dict[ticker] = df

        # 원래 df_dict를 업데이트된 버전으로 교체
        df_dict = updated_df_dict

        logger.info(f"신호 병합 완료: {len(df_dict)}개 종목")