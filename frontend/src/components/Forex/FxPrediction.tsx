"use client";
import { useEffect, useState } from "react";
import axios from "axios";
import { DateTime } from "luxon";
import formatDate from './formatDate'
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

type PredictionHistory = {
  id: number;
  date: string;
  predicted_close: number;
};

type RealData = {
  id: number;
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
};

type ForecastResponse = {
  success: boolean;
  predictions: Prediction[];
  count: number;
  type: string;
  generated_at: string;
  current_time_utc?: string;
};

function FxPrediction() {
  const [realData, setRealData] = useState<RealData[]>([]);
  const [preds, setPreds] = useState<Prediction[]>([]);
  const [predHistory, setPredHistory] = useState<PredictionHistory[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [training, setTraining] = useState<boolean>(false);
  const [fetchingData, setFetchingData] = useState<boolean>(false);
  const [currentTimeUTC, setCurrentTimeUTC] = useState<string>("");
  const [generatedAt, setGeneratedAt] = useState<string>("");
  const [predictionType, setPredictionType] = useState<string>("next");
  const [hours, setHours] = useState<number>(5);

  const fetchRealData = async() => {
    const url = 'http://localhost:8000/api/forex/real_data/'
    try {
      const res = await axios.get(url)
      console.log("res_real:", res.data);
      
      // Sort chronologically (oldest to newest)
      const sortedData = res.data
        .sort((a: RealData, b: RealData) => 
          new Date(a.date).getTime() - new Date(b.date).getTime()
        );
      
      // Only keep the last 12 hours of data
      const now = DateTime.utc();
      const twelveHoursAgo = now.minus({ hours: 12 });
      
      const recentData = sortedData.filter((p: RealData) => {
        const pointTime = DateTime.fromISO(p.date, { zone: 'utc' });
        return pointTime >= twelveHoursAgo;
      });
      
      console.log("Total real data points:", sortedData.length);
      console.log("Filtered to last 12 hours:", recentData.length);
      if (recentData.length > 0) {
        console.log("Oldest:", recentData[0].date);
        console.log("Newest:", recentData[recentData.length - 1].date);
      }
      
      setRealData(recentData)
    } catch (error) {
      console.error("Fetch real data error:", error);
      setRealData([])
      toaster.create({
        description: "Failed to fetch real data",
        type: "error",
      });
    }
  }

  const fetchPreds = async (type: string = "next", numHours: number = 1) => {
    const url = `http://localhost:8000/api/forex/forecast/?hours=${numHours}&type=${type}`;
    try {
      const res = await axios.get<ForecastResponse>(url);
      
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
    }
  };

  const fetchHistory = async() => {
    const url = 'http://127.0.0.1:8000/api/forex/predictions/history/'
    try {
      const resp = await axios.get(url)
      const history_data = resp.data
      
      // Sort by date (newest first) for display
      const sortedHistory = history_data.sort(
        (a: PredictionHistory, b: PredictionHistory) => 
          new Date(b.date).getTime() - new Date(a.date).getTime()
      );
      
      setPredHistory(sortedHistory)
    } catch(error) {
      console.error("fetch pred history failed:", error)
      toaster.create({
        description: "Failed to fetch prediction history",
        type: "error",
      });
    }
  }

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
        // Refresh predictions after training
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

  const fetchLatestData = async() => {
    const url = "http://localhost:8000/api/forex/real-data/latest-data/"
    setFetchingData(true);
    
    try {
      const response = await axios.post(url)
      console.log('📊 Full API Response:', response.data);
    console.log('📊 Latest date from API:', response.data.latest_date);
    console.log('📊 Latest price from API:', response.data.latest_price);
    console.log('📊 Total records in DB:', response.data.total_records);
      if (response.data.success) {
        console.log('✅ Data fetched successfully:', response.data);
        
        toaster.create({
          description: `Data updated! Latest price: $${response.data.latest_price?.toFixed(2)}`,
          type: "success",
        });
        
        // Refresh real data and predictions after update
        await fetchRealData();
        await fetchHistory();
        
        return response.data;
      } else {
        console.error('❌ Update data failed:', response.data.error);
        toaster.create({
          description: response.data.error || "Failed to update data",
          type: "error",
        });
      }
    } catch(error: any) {
      console.error("Refresh latest data error:", error);
      
      if (error.response?.data?.error) {
        toaster.create({
          description: error.response.data.error,
          type: "error",
        });
      } else if (error.request) {
        toaster.create({
          description: "Network error. Please check your connection.",
          type: "error",
        });
      } else {
        toaster.create({
          description: "Failed to fetch data. Please try again.",
          type: "error",
        });
      }
    } finally {
      setFetchingData(false);
    }
  }

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

  useEffect(() => {
    const fetchAllData = async () => {
      setLoading(true);
      try {
        await Promise.all([
          fetchRealData(),
          fetchPreds(),
          fetchHistory()
        ]);
      } catch (error) {
        console.error("Error fetching data:", error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchAllData();
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

  // Check if we have data to display
  const hasData = realData.length > 0 || predHistory.length > 0;

  // Combine all timestamps and sort them
  const allTimestamps = [
    ...realData.map(p => p.date),
    ...predHistory.map(p => p.date)
  ];
  
  const uniqueTimestamps = [...new Set(allTimestamps)]
    .sort((a, b) => new Date(a).getTime() - new Date(b).getTime());
  
  // Create maps for quick lookup
  const realDataMap = new Map(realData.map(p => [p.date, p.close]));
  const predHistoryMap = new Map(predHistory.map(p => [p.date, p.predicted_close]));

  // Create formatted labels for chart
  const chartLabels = uniqueTimestamps.map(ts => formatTimeEEST(ts));

  const combinedChartData = {
    labels: chartLabels,
    datasets: [
      ...(realData.length > 0 ? [{
        label: "Real Data",
        data: uniqueTimestamps.map(ts => realDataMap.get(ts) ?? null),
        borderColor: "#FFD700",
        backgroundColor: "rgba(255, 215, 0, 0.2)",
        fill: false,
        tension: 0.3,
        pointBackgroundColor: "#FFD700",
        pointBorderColor: "#B8860B",
        pointRadius: 5,
        spanGaps: false,
      }] : []),
      ...(predHistory.length > 0 ? [{
        label: "Predictions",
        data: uniqueTimestamps.map(ts => predHistoryMap.get(ts) ?? null),
        borderColor: "#e54e08ff",
        backgroundColor: "rgba(229, 78, 8, 0.2)",
        fill: false,
        tension: 0.3,
        pointBackgroundColor: "#ed4d08ff",
        pointBorderColor: "#d45320ff",
        pointRadius: 5,
        spanGaps: false,
      }] : []),
    ],
  };

  // if (combinedChartData.datasets.length > 0) {
  //   console.log("Real Data points:", combinedChartData.datasets[0]?.data.filter(d => d !== null).length);
  // }
  // if (combinedChartData.datasets.length > 1) {
  //   console.log("Prediction points:", combinedChartData.datasets[1]?.data.filter(d => d !== null).length);
  // }

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
          callback: function (value: any) {
            return "$" + value.toFixed(2);
          },
        },
      },
    },
    plugins: {
      tooltip: {
        callbacks: {
          label: (context: any) => {
            const datasetLabel = context.dataset.label;
            const value = context.parsed.y;
            return value !== null ? `${datasetLabel}: $${value.toFixed(2)}` : '';
          },
          title: (tooltipItems: any) => {
            const labelIndex = tooltipItems[0].dataIndex;
            const timestamp = uniqueTimestamps[labelIndex];
            
            if (timestamp) {
              // Find the point in either dataset
              const realPoint = realData.find(p => p.date === timestamp);
              const predPoint = predHistory.find(p => p.date === timestamp);
              const point = realPoint || predPoint;
              
              if (point) {
                return formatDateTimeEEST(point.date);
              }
            }
            
            return chartLabels[labelIndex] || 'N/A';
          },
        },
      },
      legend: {
        display: true,
        position: "top" as const,
        labels: {
          color: "#333",
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

        <Box w="full" p={4} rounded="lg" borderWidth="1px">
          <Grid templateColumns="repeat(auto-fit, minmax(200px, 1fr))" gap={4}>
            <GridItem>
              <Text fontSize="sm" color="gray.600">Current Time (EEST)</Text>
              <Text fontWeight="bold">
                {currentTimeUTC
                  ? formatDate(currentTimeUTC)
                  : formatDate(DateTime.utc().toISO())}
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
              <Text fontWeight="bold">{predHistory.length}</Text>
            </GridItem>
          </Grid>
        </Box>

        <HStack w="full" align="start" gap={6}>
          <Box flex="2" h="400px" p={4} boxShadow="md" borderWidth="1px" rounded="lg">
            {hasData ? (
              <Line data={combinedChartData} options={chartOptions} />
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
                  <Table.ColumnHeader>Type</Table.ColumnHeader>
                  <Table.ColumnHeader textAlign="end">Price</Table.ColumnHeader>
                </Table.Row>
              </Table.Header>
              <Table.Body>
                {predHistory.length > 0 ? (
                  predHistory
                  .sort((a: PredictionHistory, b: PredictionHistory) => new Date(b.date).getTime() - new Date(a.date).getTime())
                  .slice(0, 5)
                  .map((p) => (
                    <Table.Row key={`pred-${p.id}-${p.date}`}>
                      <Table.Cell>
                        <VStack align="start" gap={0}>
                          <Text fontSize="sm" fontWeight="medium">
                            {formatDate(p.date)}
                          </Text>
                        </VStack>
                      </Table.Cell>
                      <Table.Cell>
                        <Badge size="sm" colorScheme="orange">
                          Prediction
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
          <Button
            onClick={fetchLatestData}
            isLoading={fetchingData}
            loadingText="Updating..."
            colorScheme="cyan"
            size="sm"
          >
            Update Latest Data
          </Button>
        </HStack>

        {generatedAt && (
          <Text fontSize="xs" color="gray.500" textAlign="center">
            Last updated: {formatDate(generatedAt)}
          </Text>
        )}
      </VStack>
    </Box>
  );
}

export default FxPrediction;