import { useEffect, useState } from "react";
import axios from "axios";
import { DateTime } from "luxon";
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
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
} from "@chakra-ui/react";
import { toast } from "react-hot-toast";
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

// Type definitions for API responses
interface Prediction {
  datetime: string;
  hour: number;
  predicted_close: number;
  time_from_now?: string;
}

interface ForecastResponse {
  success: boolean;
  predictions: Prediction[];
  count: number;
  type: string;
  generated_at: string;
  current_time_utc?: string;
}

interface DataStatusResponse {
  success: boolean;
  total_records: number;
  latest_date: string | null;
  oldest_date: string | null;
  hours_covered: number;
  latest_price: number | null;
  current_time: string;
  error?: string;
}

interface RecentDataPoint {
  datetime: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

interface RecentDataResponse {
  success: boolean;
  data: RecentDataPoint[];
  count: number;
  hours_requested: number;
  error?: string;
}

interface TrainResponse {
  success: boolean;
  status: string;
  message: string;
}

interface FetchDataResponse {
  success: boolean;
  message: string;
  total_records: number;
  latest_date: string | null;
  latest_price: number | null;
  error?: string;
}

// Timezone formatting functions
const formatDateTimeEEST = (isoString: string): string => {
  if (!isoString) return "N/A";
  return DateTime.fromISO(isoString, { zone: "utc" })
    .setZone("Europe/Helsinki")
    .toFormat("hh:mm a 'EEST on' cccc, LLLL d, yyyy");
};

const formatTimeEEST = (isoString: string): string => {
  if (!isoString) return "N/A";
  return DateTime.fromISO(isoString, { zone: "utc" })
    .setZone("Europe/Helsinki")
    .toFormat("hh:mm a");
};

const formatDateEEST = (isoString: string): string => {
  if (!isoString) return "N/A";
  return DateTime.fromISO(isoString, { zone: "utc" })
    .setZone("Europe/Helsinki")
    .toFormat("cccc, LLLL d, yyyy");
};

const FxDashboard: React.FC = () => {
  // State for predictions
  const [preds, setPreds] = useState<Prediction[]>([]);
  const [loadingPreds, setLoadingPreds] = useState<boolean>(true);
  const [predictionType, setPredictionType] = useState<string>("next");
  const [hours, setHours] = useState<number>(5);
  const [generatedAt, setGeneratedAt] = useState<string>("");
  const [currentTimeUTC, setCurrentTimeUTC] = useState<string>("");

  // State for data status
  const [dataStatus, setDataStatus] = useState<DataStatusResponse | null>(null);
  const [loadingStatus, setLoadingStatus] = useState<boolean>(true);

  // State for recent data
  const [recentData, setRecentData] = useState<RecentDataPoint[]>([]);
  const [loadingRecentData, setLoadingRecentData] = useState<boolean>(true);

  // State for training and fetching
  const [training, setTraining] = useState<boolean>(false);
  const [fetchingData, setFetchingData] = useState<boolean>(false);

  // Fetch predictions
  const fetchPreds = async (type: string = "next", numHours: number = 5) => {
    setLoadingPreds(true);
    const currentTimeUTC = DateTime.utc().toISO();
    const url = `http://localhost:8000/api/forex/forecast/?hours=${numHours}&type=${type}&current_time_utc=${encodeURIComponent(currentTimeUTC || "")}`;

    try {
      const res = await axios.get<ForecastResponse>(url);
      console.log("Forecast Response:", res.data);

      if (res.data.success) {
        setPreds(res.data.predictions);
        setGeneratedAt(res.data.generated_at);
        setCurrentTimeUTC(res.data.current_time_utc || currentTimeUTC || DateTime.utc().toISO());
        setPredictionType(res.data.type);
        setHours(numHours);
        toast.success(`Successfully loaded ${res.data.count} predictions`);
      } else {
        throw new Error(res.data.message || "Failed to fetch predictions");
      }
    } catch (error: any) {
      console.error("Fetch predictions error:", error);
      setPreds([]);
      toast.error(error.response?.data?.message || "Failed to fetch predictions. Please try training the model first.");
    } finally {
      setLoadingPreds(false);
    }
  };

  // Fetch data status
  const fetchDataStatus = async () => {
    setLoadingStatus(true);
    try {
      const res = await axios.get<DataStatusResponse>("http://localhost:8000/api/forex/data-status/");
      console.log("Data Status Response:", res.data);

      if (res.data.success) {
        setDataStatus(res.data);
        toast.success("Successfully loaded data status");
      } else {
        throw new Error(res.data.error || "Failed to fetch data status");
      }
    } catch (error: any) {
      console.error("Fetch data status error:", error);
      setDataStatus(null);
      toast.error(error.response?.data?.error || "Failed to fetch data status");
    } finally {
      setLoadingStatus(false);
    }
  };

  // Fetch recent data
  const fetchRecentData = async (hours: number = 24) => {
    setLoadingRecentData(true);
    try {
      const res = await axios.get<RecentDataResponse>(`http://localhost:8000/api/forex/recent-data/?hours=${hours}`);
      console.log("Recent Data Response:", res.data);

      if (res.data.success) {
        setRecentData(res.data.data);
        toast.success(`Successfully loaded ${res.data.count} recent data points`);
      } else {
        throw new Error(res.data.error || "Failed to fetch recent data");
      }
    } catch (error: any) {
      console.error("Fetch recent data error:", error);
      setRecentData([]);
      toast.error(error.response?.data?.error || "Failed to fetch recent data");
    } finally {
      setLoadingRecentData(false);
    }
  };

  // Train model
  const trainModel = async () => {
    setTraining(true);
    try {
      const res = await axios.post<TrainResponse>("http://localhost:8000/api/forex/train/", { epochs: 500 });
      console.log("Train Response:", res.data);

      if (res.data.success) {
        toast.success(res.data.message);
        setTimeout(() => fetchPreds(predictionType, hours), 1000);
      } else {
        throw new Error(res.data.message || "Training failed");
      }
    } catch (error: any) {
      console.error("Training error:", error);
      toast.error(error.response?.data?.message || "Training failed");
    } finally {
      setTraining(false);
    }
  };

  // Fetch latest data
  const fetchLatestData = async () => {
    setFetchingData(true);
    try {
      const res = await axios.post<FetchDataResponse>("http://localhost:8000/api/forex/fetch-data/");
      console.log("Fetch Data Response:", res.data);

      if (res.data.success) {
        toast.success(res.data.message);
        fetchDataStatus();
        fetchRecentData();
      } else {
        throw new Error(res.data.error || "Failed to fetch latest data");
      }
    } catch (error: any) {
      console.error("Fetch latest data error:", error);
      toast.error(error.response?.data?.error || "Failed to fetch latest data");
    } finally {
      setFetchingData(false);
    }
  };

  // Initial data fetch
  useEffect(() => {
    fetchPreds();
    fetchDataStatus();
    fetchRecentData();
  }, []);

  // Chart configuration
  const chartData = {
    labels: preds.length > 0 ? preds.map((p) => formatTimeEEST(p.datetime)) : [],
    datasets: [
      {
        label: "Predicted Close Price (USD)",
        data: preds.length > 0 ? preds.map((p) => p.predicted_close) : [],
        borderColor: "#FFD700",
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
        grid: { display: true, color: "rgba(255, 255, 255, 0.1)" },
      },
      y: {
        title: { display: true, text: "Price (USD)" },
        grid: { display: true, color: "rgba(255, 255, 255, 0.1)" },
        ticks: {
          callback: (value: number) => `$${value.toFixed(2)}`,
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
        labels: { color: "#333" },
      },
    },
  };

  // Loading state
  if (loadingPreds || loadingStatus || loadingRecentData || training || fetchingData) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minH="50vh">
        <VStack>
          <Spinner size="xl" />
          <Text>
            {training
              ? "Training model..."
              : fetchingData
              ? "Fetching latest data..."
              : "Loading data..."}
          </Text>
        </VStack>
      </Box>
    );
  }

  return (
    <Box mx="auto" py={10} px={4} maxW="7xl">
      <VStack gap={8}>
        <Heading size="lg" textAlign="center">
          XAU/USD Forex Dashboard
        </Heading>

        {/* Tabs for Predictions, Recent Data, and Status */}
        <Tabs variant="soft-rounded" colorScheme="blue">
          <TabList>
            <Tab>Predictions</Tab>
            <Tab>Recent Data</Tab>
            <Tab>Data Status</Tab>
          </TabList>

          <TabPanels>
            {/* Predictions Tab */}
            <TabPanel>
              <VStack gap={6}>
                {/* Status Information */}
                <Box w="full" p={4} bg="gray.50" rounded="lg" borderWidth="1px">
                  <Grid templateColumns="repeat(auto-fit, minmax(200px, 1fr))" gap={4}>
                    <GridItem>
                      <Text fontSize="sm" color="gray.600">Current Time (EEST)</Text>
                      <Text fontWeight="bold">
                        {currentTimeUTC ? formatDateTimeEEST(currentTimeUTC) : "N/A"}
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

                {/* Predictions Chart and Table */}
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
                            Try training the model or fetching data
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
                                    {formatTimeEEST(p.datetime)}
                                  </Text>
                                  <Text fontSize="xs" color="gray.500">
                                    {formatDateEEST(p.datetime)}
                                  </Text>
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
                                  Click "Train Model" or "Fetch Data" to get started
                                </Text>
                              </VStack>
                            </Table.Cell>
                          </Table.Row>
                        )}
                      </Table.Body>
                    </Table.Root>
                  </Box>
                </HStack>
              </VStack>
            </TabPanel>

            {/* Recent Data Tab */}
            <TabPanel>
              <Box w="full" p={4} boxShadow="md" borderWidth="1px" rounded="lg">
                <Table.Root size="sm">
                  <Table.Header>
                    <Table.Row>
                      <Table.ColumnHeader>Time (EEST)</Table.ColumnHeader>
                      <Table.ColumnHeader>Open</Table.ColumnHeader>
                      <Table.ColumnHeader>High</Table.ColumnHeader>
                      <Table.ColumnHeader>Low</Table.ColumnHeader>
                      <Table.ColumnHeader>Close</Table.ColumnHeader>
                    </Table.Row>
                  </Table.Header>
                  <Table.Body>
                    {recentData.length > 0 ? (
                      recentData.map((d, i) => (
                        <Table.Row key={i}>
                          <Table.Cell>
                            <VStack align="start" gap={0}>
                              <Text fontSize="sm" fontWeight="medium">
                                {formatTimeEEST(d.datetime)}
                              </Text>
                              <Text fontSize="xs" color="gray.500">
                                {formatDateEEST(d.datetime)}
                              </Text>
                            </VStack>
                          </Table.Cell>
                          <Table.Cell>${d.open.toFixed(2)}</Table.Cell>
                          <Table.Cell>${d.high.toFixed(2)}</Table.Cell>
                          <Table.Cell>${d.low.toFixed(2)}</Table.Cell>
                          <Table.Cell fontWeight="bold">${d.close.toFixed(2)}</Table.Cell>
                        </Table.Row>
                      ))
                    ) : (
                      <Table.Row>
                        <Table.Cell colSpan={5} textAlign="center" py={8}>
                          <VStack>
                            <Text>No recent data available</Text>
                            <Text fontSize="sm" color="gray.500">
                              Click "Fetch Data" to get started
                            </Text>
                          </VStack>
                        </Table.Row>
                      )}
                    </Table.Body>
                  </Table.Root>
                </Box>
              </TabPanel>

              {/* Data Status Tab */}
              <TabPanel>
                <Box w="full" p={4} bg="gray.50" rounded="lg" borderWidth="1px">
                  {dataStatus ? (
                    <Grid templateColumns="repeat(auto-fit, minmax(200px, 1fr))" gap={4}>
                      <GridItem>
                        <Text fontSize="sm" color="gray.600">Total Records</Text>
                        <Text fontWeight="bold">{dataStatus.total_records}</Text>
                      </GridItem>
                      <GridItem>
                        <Text fontSize="sm" color="gray.600">Latest Data</Text>
                        <Text fontWeight="bold">
                          {dataStatus.latest_date ? formatDateTimeEEST(dataStatus.latest_date) : "N/A"}
                        </Text>
                      </GridItem>
                      <GridItem>
                        <Text fontSize="sm" color="gray.600">Oldest Data</Text>
                        <Text fontWeight="bold">
                          {dataStatus.oldest_date ? formatDateTimeEEST(dataStatus.oldest_date) : "N/A"}
                        </Text>
                      </GridItem>
                      <GridItem>
                        <Text fontSize="sm" color="gray.600">Hours Covered</Text>
                        <Text fontWeight="bold">{dataStatus.hours_covered}</Text>
                      </GridItem>
                      <GridItem>
                        <Text fontSize="sm" color="gray.600">Latest Price</Text>
                        <Text fontWeight="bold">
                          {dataStatus.latest_price ? `$${dataStatus.latest_price.toFixed(2)}` : "N/A"}
                        </Text>
                      </GridItem>
                      <GridItem>
                        <Text fontSize="sm" color="gray.600">Current Time (EEST)</Text>
                        <Text fontWeight="bold">
                          {dataStatus.current_time ? formatDateTimeEEST(dataStatus.current_time) : "N/A"}
                        </Text>
                      </GridItem>
                    </Grid>
                  ) : (
                    <Text textAlign="center">No data status available</Text>
                  )}
                </Box>
              </TabPanel>
            </TabPanels>
          </Tabs>

          {/* Action Buttons */}
          <HStack gap={4} flexWrap="wrap">
            <Button
              onClick={() => fetchPreds("current", 1)}
              isLoading={loadingPreds}
              loadingText="Loading"
              colorScheme="green"
              size="sm"
            >
              Current Hour
            </Button>
            <Button
              onClick={() => fetchPreds("next", 5)}
              isLoading={loadingPreds}
              loadingText="Loading"
              colorScheme="blue"
              size="sm"
            >
              Next 5 Hours
            </Button>
            <Button
              onClick={() => fetchPreds("next", 24)}
              isLoading={loadingPreds}
              loadingText="Loading"
              colorScheme="purple"
              size="sm"
            >
              Next 24 Hours
            </Button>
            <Button
              onClick={() => fetchPreds("today", 0)}
              isLoading={loadingPreds}
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
            <Button
              onClick={fetchLatestData}
              isLoading={fetchingData}
              loadingText="Fetching..."
              colorScheme="cyan"
              size="sm"
            >
              Fetch Latest Data
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
};

export default FxDashboard;