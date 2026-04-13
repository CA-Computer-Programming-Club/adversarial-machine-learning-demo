import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import {
  Alert,
  AppBar,
  Box,
  Button,
  Card,
  CardContent,
  Container,
  Divider,
  FormControl,
  FormControlLabel,
  InputLabel,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  MenuItem,
  Paper,
  Radio,
  RadioGroup,
  Select,
  Slider,
  Stack,
  Toolbar,
  Typography,
} from "@mui/material";

declare global {
  interface Window {
    Chart: any;
  }
}

type CorpusImage = { name: string; url: string };
type TopPrediction = { id: string; label: string; confidence: number };
type AnalyzeResponse = {
  baseImage: string;
  perturbedImage: string;
  baseTop: TopPrediction[];
  perturbedTop: TopPrediction[];
  epsilon: number;
  steps: number;
  alpha: number;
  attack: string;
};

const API = "https://math-expo.cacpc.dev/api";

export default function App() {
  const [corpus, setCorpus] = useState<CorpusImage[]>([]);
  const [inputMode, setInputMode] = useState<"preset" | "upload">("preset");
  const [selectedCorpus, setSelectedCorpus] = useState("labrador.jpg");
  const [uploadedImage, setUploadedImage] = useState<string>("");
  const [attack, setAttack] = useState("fgsm");
  const [epsilon, setEpsilon] = useState(0.1);
  const [steps, setSteps] = useState(5);
  const [alpha, setAlpha] = useState(0.02);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [uploadMessage, setUploadMessage] = useState("");
  const chartRef = useRef<HTMLCanvasElement | null>(null);
  const chartInstanceRef = useRef<any>(null);

  useEffect(() => {
    fetch(`${API}/corpus`)
      .then((r) => r.json())
      .then((data) => {
        setCorpus(data.images);
        if (data.images?.length && !selectedCorpus) {
          setSelectedCorpus(data.images[0].name);
        }
      });
  }, []);

  const activeSourceLabel = useMemo(
    () => (inputMode === "preset" ? selectedCorpus : "Uploaded image"),
    [inputMode, selectedCorpus],
  );

  const chartRows = useMemo(() => {
    if (!result) return [] as { label: string; confidence: number }[];
    const baseTop = result.baseTop[0];
    const perturbedTop = result.perturbedTop[0];
    return [
      { label: `Original: ${baseTop.label}`, confidence: baseTop.confidence },
      {
        label: `Perturbed: ${perturbedTop.label}`,
        confidence: perturbedTop.confidence,
      },
    ];
  }, [result]);

  const analyze = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          corpusImage: inputMode === "preset" ? selectedCorpus : undefined,
          image:
            inputMode === "upload" ? uploadedImage || undefined : undefined,
          attack,
          epsilon,
          steps,
          alpha,
        }),
      });
      if (!response.ok) throw new Error("Backend request failed");
      const data = (await response.json()) as AnalyzeResponse;
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (corpus.length) void analyze();
  }, [corpus]);

  useEffect(() => {
    if (!result || !chartRef.current || !window.Chart) return;
    chartInstanceRef.current?.destroy?.();
    chartInstanceRef.current = new window.Chart(chartRef.current, {
      type: "bar",
      // disable legend
      data: {
        labels: chartRows.map((item) => item.label),
        datasets: [
          {
            label: "Prediction confidence",
            data: chartRows.map((x) => Number((x.confidence * 100).toFixed(1))),
            backgroundColor: ["rgba(25,118,210,0.8)", "rgba(211,47,47,0.75)"],
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { autoSkip: false, maxRotation: 25, minRotation: 0 } },
          y: {
            beginAtZero: true,
            max: 100,
            title: { display: true, text: "Confidence (%)" },
          },
        },
      },
    });
    return () => chartInstanceRef.current?.destroy?.();
  }, [result, chartRows]);

  const loadFile = (file: File) => {
    if (!file.type.startsWith("image/")) {
      setUploadMessage("That file is not an image.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      setUploadedImage(String(reader.result || ""));
      setInputMode("upload");
      setUploadMessage(`Loaded image: ${file.name}`);
    };
    reader.onerror = () => {
      setUploadMessage("Failed to read that image file.");
    };
    reader.readAsDataURL(file);
  };

  const onUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    loadFile(file);
  };

  const onPaste = async () => {
    try {
      if (!navigator.clipboard || !("read" in navigator.clipboard)) {
        setUploadMessage("Clipboard image paste is not supported in this browser.");
        return;
      }
      const items = await (navigator.clipboard as Clipboard & { read: () => Promise<ClipboardItem[]> }).read();
      for (const item of items) {
        const imageType = item.types.find((type) => type.startsWith("image/"));
        if (!imageType) continue;
        const blob = await item.getType(imageType);
        const file = new File([blob], `pasted-image.${imageType.split("/")[1] || "png"}`, { type: imageType });
        loadFile(file);
        return;
      }
      setUploadMessage("No image was found in the clipboard.");
    } catch (err) {
      setUploadMessage(err instanceof Error ? err.message : "Failed to read image from clipboard.");
    }
  };

  return (
    <Box sx={{ minHeight: "100vh", bgcolor: "#f5f5f5" }}>
      <AppBar position="static" color="primary">
        <Toolbar sx={{ overflow: "hidden" }}>
          <Typography
            variant="h6"
            sx={{
              fontWeight: 600,
              whiteSpace: "nowrap",
              flexShrink: 0,
            }}
          >
            CA Math Expo
          </Typography>

          <Box
            sx={{
              width: 2,
              height: 25,
              bgcolor: "rgb(255,255,255)",
              mx: 2,
              flexShrink: 0,
            }}
          />

          <Typography
            variant="h6"
            sx={{
              fontWeight: 500,
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
              minWidth: 0,
            }}
          >
            Adversarial Machine Learning Attacks
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ py: 4 }}>
        <Stack spacing={3}>
          <Paper elevation={1} sx={{ p: 3 }}>
            <Typography variant="h4" gutterBottom>
              Adversarial Image Playground
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Run adversarial perturbations and compare how the model's
              predictions change.
            </Typography>
          </Paper>

          <Stack
            direction={{ xs: "column", md: "row" }}
            spacing={3}
            sx={{ alignItems: "stretch" }}
          >
            <Card sx={{ width: { xs: "100%", md: 380 }, flexShrink: 0 }}>
              <CardContent>
                <Stack spacing={3}>
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      Input source
                    </Typography>
                    <RadioGroup
                      value={inputMode}
                      onChange={(e) =>
                        setInputMode(e.target.value as "preset" | "upload")
                      }
                    >
                      <FormControlLabel
                        value="preset"
                        control={<Radio />}
                        label="Use a preset image"
                      />
                      <FormControlLabel
                        value="upload"
                        control={<Radio />}
                        label="Upload your own image"
                      />
                    </RadioGroup>
                  </Box>

                  {inputMode === "preset" ? (
                    <FormControl fullWidth>
                      <InputLabel id="corpus-label">Preset image</InputLabel>
                      <Select
                        labelId="corpus-label"
                        label="Preset image"
                        value={selectedCorpus}
                        onChange={(e) =>
                          setSelectedCorpus(String(e.target.value))
                        }
                      >
                        {corpus.map((item) => (
                          <MenuItem key={item.name} value={item.name}>
                            {item.name}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  ) : (
                    <Stack spacing={1.5}>
                      <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5}>
                        <Button variant="outlined" component="label">
                          Upload image
                          <input
                            hidden
                            type="file"
                            accept="image/*"
                            onChange={onUpload}
                          />
                        </Button>
                        <Button variant="outlined" onClick={() => void onPaste()}>
                          Paste image
                        </Button>
                      </Stack>
                      <Typography
                        variant="body2"
                        color="text.secondary"
                        sx={{ mt: 1 }}
                      >
                        {uploadedImage
                          ? "Custom image loaded."
                          : "No uploaded or pasted image selected yet."}
                      </Typography>
                      {uploadMessage ? <Alert severity="info">{uploadMessage}</Alert> : null}
                    </Stack>
                  )}

                  <Divider />

                  <FormControl fullWidth>
                    <InputLabel id="attack-label">Attack type</InputLabel>
                    <Select
                      labelId="attack-label"
                      label="Attack type"
                      value={attack}
                      onChange={(e) => setAttack(String(e.target.value))}
                    >
                      <MenuItem value="fgsm">FGSM</MenuItem>
                      <MenuItem value="iterative">Iterative attack</MenuItem>
                    </Select>
                  </FormControl>

                  <Box>
                    <Typography gutterBottom>
                      Epsilon: {epsilon.toFixed(2)}
                    </Typography>
                    <Slider
                      min={0.01}
                      max={0.3}
                      step={0.01}
                      value={epsilon}
                      onChange={(_, v) => setEpsilon(v as number)}
                    />
                    <Typography variant="body2" color="text.secondary">
                      Epsilon controls the maximum overall perturbation size.
                      Higher epsilon usually makes the attack stronger and more
                      visible.
                    </Typography>
                  </Box>
                  <Box>
                    <Typography gutterBottom>Steps: {steps}</Typography>
                    <Slider
                      min={1}
                      max={20}
                      step={1}
                      value={steps}
                      onChange={(_, v) => setSteps(v as number)}
                    />
                    <Typography variant="body2" color="text.secondary">
                      Steps only matter for the iterative attack. More steps
                      means the attack is applied in several smaller moves.
                    </Typography>
                  </Box>
                  <Box>
                    <Typography gutterBottom>
                      Alpha: {alpha.toFixed(3)}
                    </Typography>
                    <Slider
                      min={0.005}
                      max={0.08}
                      step={0.005}
                      value={alpha}
                      onChange={(_, v) => setAlpha(v as number)}
                    />
                    <Typography variant="body2" color="text.secondary">
                      Alpha is the size of each iterative step.
                    </Typography>
                  </Box>

                  <Button
                    variant="contained"
                    size="large"
                    onClick={() => void analyze()}
                    disabled={loading}
                  >
                    Run attack
                  </Button>
                  {loading ? <LinearProgress /> : null}
                  {error ? <Alert severity="error">{error}</Alert> : null}
                </Stack>
              </CardContent>
            </Card>

            <Stack spacing={3} sx={{ flex: 1 }}>
              <Stack direction={{ xs: "column", lg: "row" }} spacing={3}>
                <Card sx={{ flex: 1 }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Original image
                    </Typography>
                    {result ? (
                      <Box
                        component="img"
                        src={result.baseImage}
                        alt="Original"
                        sx={{
                          width: "100%",
                          borderRadius: 2,
                          border: "1px solid #ddd",
                        }}
                      />
                    ) : null}
                    <Typography variant="subtitle1" sx={{ mt: 2, mb: 1 }}>
                      Top 5 model predictions for the original image
                    </Typography>
                    <List dense>
                      {result?.baseTop.map((item, idx) => (
                        <ListItem key={item.id} divider>
                          <ListItemText
                            primary={`${idx + 1}. ${item.label}`}
                            secondary={`Confidence: ${(item.confidence * 100).toFixed(2)}%`}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </CardContent>
                </Card>

                <Card sx={{ flex: 1 }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Perturbed image
                    </Typography>
                    {result ? (
                      <Box
                        component="img"
                        src={result.perturbedImage}
                        alt="Perturbed"
                        sx={{
                          width: "100%",
                          borderRadius: 2,
                          border: "1px solid #ddd",
                        }}
                      />
                    ) : null}
                    <Typography variant="subtitle1" sx={{ mt: 2, mb: 1 }}>
                      Top 5 model predictions for the perturbed (modified) image
                    </Typography>
                    <List dense>
                      {result?.perturbedTop.map((item, idx) => (
                        <ListItem key={item.id} divider>
                          <ListItemText
                            primary={`${idx + 1}. ${item.label}`}
                            secondary={`Confidence: ${(item.confidence * 100).toFixed(2)}%`}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </CardContent>
                </Card>
              </Stack>

              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Prediction confidence
                  </Typography>
                  <Box sx={{ height: 360 }}>
                    <canvas ref={chartRef} />
                  </Box>
                </CardContent>
              </Card>
            </Stack>
          </Stack>
        </Stack>
      </Container>
    </Box>
  );
}
