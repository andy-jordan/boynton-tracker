// Shape of the model's summary JSON. Fields are optional where the model may
// not emit them yet, so the page degrades instead of breaking.
export interface Benchmark {
  key: string;
  label?: string;
  p: number;
  status?: string;
}

export interface LeverageGame {
  date: string;
  opponent: string;
  site?: string; // H, A or N
  p_win: number;
  p_ext_if_win: number;
  p_ext_if_loss: number;
}

export interface Summary {
  source?: 'kenpom' | 'torvik';
  sample_data?: boolean;
  generated_at?: string;
  as_of?: string; // ratings date, or "illustrative" when the sample ratings were used
  n_sims?: number;
  record?: { wins: number; losses: number };
  hits_dist?: Record<string, number>;
  reg_wins_mean?: number;
  p_extension: number;
  benchmarks: Benchmark[] | Record<string, number | { p: number; label?: string; status?: string }>;
  leverage?: LeverageGame[];
}

export interface HistoryPoint {
  source?: 'kenpom' | 'torvik';
  date: string;
  p_extension: number;
  note?: string;
}
