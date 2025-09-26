"use client";
import { useEffect, useState } from "react";
import axios from "axios";
import { DateTime } from "luxon";
import formatDate from '../components/Forex/formatDate'
import {
  Box,
  Heading,
  Spinner,
  Table,
  VStack,
  HStack,
  Button,
  Text,
  Badge,
  Grid,
  GridItem,
} from "@chakra-ui/react";
import { toaster } from "@/components/ui/toaster";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
  Legend,
  Filler
);

type Prediction = {
  datetime: string;
  hour: number;
  predicted_close: number;
  time_from_now?: string;
};

type ForecastResponse = {
  success: boolean;
  predictions: Prediction[];
  count: number;
  type: string;
  generated_at: string;
  current_time_utc?: string;
};

function FxDashboard() {
  const [preds, setPreds] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [training, setTraining] = useState<boolean>(false);
  const [currentTimeUTC, setCurrentTimeUTC] = useState<string>("");
  const [generatedAt, setGeneratedAt] = useState<string>("");
  const [predictionType, setPredictionType] = useState<string>("next");
  const [hours, setHours] = useState<number>(5);

  const fetchPreds = async (type: string = "next", numHours: number = 5) => {
    setLoading(true);
    const url = `http://localhost:8000/api/forex/forecast/?hours=${numHours}&type=${type}`;

    try {
      const res = await axios.get<ForecastResponse>(url);
      console.log("res:", res.data);

      if (res.data.success) {
        setPreds(res.data.predictions);
        setGeneratedAt(res.data.generated_at);
        setCurrentTimeUTC(res.data.current_time_utc || DateTime.utc().toISO());
        setPredictionType(res.data.type);

        toaster.create({
          description: `Successfully loaded ${res.data.count} predictions`,
          type: "success",
        });
      } else {
        throw new Error("Failed to fetch predictions");
      }
    } catch (error) {
      console.error("Fetch predictions error:", error);
      setPreds([]);
      toaster.create({
        description: "Failed to fetch predictions. Please try training the model first.",
        type: "error",
      });
    } finally {
      setLoading(false);
    }
  };

  const trainModel = async () => {
    setTraining(true);
    const url = "http://localhost:8000/api/forex/train/";

    try {
      const res = await axios.post(url, { epochs: 500 });

      if (res.data.success) {
        toaster.create({
          description: res.data.message || "Model trained successfully",
          type: "success",
        });

        // Automatically fetch predictions after training
        setTimeout(() => fetchPreds(predictionType, hours), 1000);
      } else {
        throw new Error(res.data.message || "Training failed");
      }
    } catch (error: any) {
      console.error("Training error:", error);
      toaster.create({
        description: error.response?.data?.message || "Training failed",
        type: "error",
      });
    } finally {
      setTraining(false);
    }
  };

  const formatDateTimeEEST = (isoString: string) => {
    if (!isoString) return "N/A";
    return DateTime.fromISO(isoString, { zone: "utc" })
      .setZone("Europe/Helsinki")
      .toFormat("hh:mm a 'EEST on' cccc, LLLL d, yyyy");
  };

  const formatTimeEEST = (isoString: string) => {
    if (!isoString) return "N/A";
    return DateTime.fromISO(isoString, { zone: "utc" })
      .setZone("Europe/Helsinki")
      .toFormat("hh:mm a");
  };

  const formatDateEEST = (isoString: string) => {
    if (!isoString) return "N/A";
    return DateTime.fromISO(isoString, { zone: "utc" })
      .setZone("Europe/Helsinki")
      .toFormat("cccc, LLLL d, yyyy");
  };

  useEffect(() => {
    fetchPreds();
  }, []);

  if (loading || training) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minH="50vh">
        <VStack>
          <Spinner size="xl" />
          <Text>{training ? "Training model..." : "Loading predictions..."}</Text>
        </VStack>
      </Box>
    );
  }

  const chartData = {
    labels: preds.length > 0 ? preds.map((p) => formatTimeEEST(p.datetime)) : [],
    datasets: [
      {
        label: "Predicted Close Price (USD)",
        data: preds.length > 0 ? preds.map((p) => p.predicted_close) : [],
        borderColor: "#FFD700", // Gold for better visibility
        backgroundColor: "rgba(255, 215, 0, 0.2)",
        fill: true,
        tension: 0.3,
        pointBackgroundColor: "#FFD700",
        pointBorderColor: "#B8860B",
        pointRadius: 5,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        title: { display: true, text: "Time (EEST)" },
        grid: { display: true, color: "rgba(255, 255, 255, 0.1)" }, // Light grid for dark/light themes
      },
      y: {
        title: { display: true, text: "Price (USD)" },
        grid: { display: true, color: "rgba(255, 255, 255, 0.1)" },
        ticks: {
          callback: function (value: any) {
            return "$" + value.toFixed(2);
          },
        },
      },
    },
    plugins: {
      tooltip: {
        callbacks: {
          label: (context: any) => `Price: $${context.parsed.y.toFixed(2)}`,
          title: (tooltipItems: any) => {
            const pred = preds[tooltipItems[0].dataIndex];
            return `${formatDateTimeEEST(pred.datetime)}${
              pred.time_from_now ? ` (${pred.time_from_now})` : ""
            }`;
          },
        },
      },
      legend: {
        display: true,
        position: "top" as const,
        labels: {
          color: "#333", // Better contrast for both themes
        },
      },
    },
  };

  return (
    <Box mx="auto" py={10} px={4} maxW="7xl">
      <VStack gap={8}>
        <Heading size="lg" textAlign="center">
          XAU/USD Forex Predictions
        </Heading>

        {/* Status Information */}
        <Box w="full" p={4} bg="gray.50" rounded="lg" borderWidth="1px">
          <Grid templateColumns="repeat(auto-fit, minmax(200px, 1fr))" gap={4}>
            <GridItem>
              <Text fontSize="sm" color="gray.600">Current Time (EEST)</Text>
              <Text fontWeight="bold">
                {currentTimeUTC
                  ? formatDateTimeEEST(currentTimeUTC)
                  : formatDateTimeEEST(DateTime.utc().toISO())}
              </Text>
            </GridItem>
            <GridItem>
              <Text fontSize="sm" color="gray.600">Generated At</Text>
              <Text fontWeight="bold">
                {generatedAt ? formatDateTimeEEST(generatedAt) : "N/A"}
              </Text>
            </GridItem>
            <GridItem>
              <Text fontSize="sm" color="gray.600">Prediction Type</Text>
              <Badge
                colorScheme={
                  predictionType === "current"
                    ? "green"
                    : predictionType === "today"
                    ? "blue"
                    : "purple"
                }
              >
                {predictionType.toUpperCase()}
              </Badge>
            </GridItem>
            <GridItem>
              <Text fontSize="sm" color="gray.600">Total Predictions</Text>
              <Text fontWeight="bold">{preds.length}</Text>
            </GridItem>
          </Grid>
        </Box>

        {/* Chart and Table */}
        <HStack w="full" align="start" gap={6}>
          <Box flex="2" h="400px" p={4} boxShadow="md" borderWidth="1px" rounded="lg">
            {preds.length > 0 ? (
              <Line data={chartData} options={chartOptions} />
            ) : (
              <Box
                textAlign="center"
                h="full"
                display="flex"
                alignItems="center"
                justifyContent="center"
              >
                <VStack>
                  <Text fontSize="lg">No predictions available</Text>
                  <Text fontSize="sm" color="gray.500">
                    Try training the model first
                  </Text>
                </VStack>
              </Box>
            )}
          </Box>

          <Box
            flex="1"
            maxH="400px"
            overflowY="auto"
            boxShadow="md"
            borderWidth="1px"
            rounded="lg"
            p={4}
          >
            <Table.Root size="sm">
              <Table.Header>
                <Table.Row>
                  <Table.ColumnHeader>Time (EEST)</Table.ColumnHeader>
                  <Table.ColumnHeader>From Now</Table.ColumnHeader>
                  <Table.ColumnHeader textAlign="end">Price</Table.ColumnHeader>
                </Table.Row>
              </Table.Header>
              <Table.Body>
                {preds.length > 0 ? (
                  preds.map((p, i) => (
                    <Table.Row key={i}>
                      <Table.Cell>
                        <VStack align="start" gap={0}>
                          <Text fontSize="sm" fontWeight="medium">
                            {formatDate(p.datetime)}
                          </Text>
                          {/* <Text fontSize="xs" color="gray.500">
                            {formatDate(p.datetime)}
                          </Text> */}
                        </VStack>
                      </Table.Cell>
                      <Table.Cell>
                        <Badge size="sm" colorScheme="blue">
                          {p.time_from_now || `+${i + 1}h`}
                        </Badge>
                      </Table.Cell>
                      <Table.Cell textAlign="end" fontWeight="bold">
                        ${p.predicted_close.toFixed(2)}
                      </Table.Cell>
                    </Table.Row>
                  ))
                ) : (
                  <Table.Row>
                    <Table.Cell colSpan={3} textAlign="center" py={8}>
                      <VStack>
                        <Text>No predictions available</Text>
                        <Text fontSize="sm" color="gray.500">
                          Click "Train Model" to get started
                        </Text>
                      </VStack>
                    </Table.Cell>
                  </Table.Row>
                )}
              </Table.Body>
            </Table.Root>
          </Box>
        </HStack>

        {/* Action Buttons */}
        <HStack gap={4} flexWrap="wrap">
          <Button
            onClick={() => fetchPreds("current", 1)}
            isLoading={loading}
            loadingText="Loading"
            colorScheme="green"
            size="sm"
          >
            Current Hour
          </Button>

          <Button
            onClick={() => fetchPreds("next", 5)}
            isLoading={loading}
            loadingText="Loading"
            colorScheme="blue"
            size="sm"
          >
            Next 5 Hours
          </Button>

          <Button
            onClick={() => fetchPreds("next", 24)}
            isLoading={loading}
            loadingText="Loading"
            colorScheme="purple"
            size="sm"
          >
            Next 24 Hours
          </Button>

          <Button
            onClick={() => fetchPreds("today", 0)}
            isLoading={loading}
            loadingText="Loading"
            colorScheme="teal"
            size="sm"
          >
            Rest of Today
          </Button>

          <Button
            onClick={trainModel}
            isLoading={training}
            loadingText="Training..."
            colorScheme="orange"
            size="sm"
          >
            Train Model
          </Button>
        </HStack>

        {/* Last Update Info */}
        {generatedAt && (
          <Text fontSize="xs" color="gray.500" textAlign="center">
            Last updated: {formatDateTimeEEST(generatedAt)}
          </Text>
        )}
      </VStack>
    </Box>
  );
}

export default FxDashboard;
