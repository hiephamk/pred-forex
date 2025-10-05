import { Box, HStack, Text, Table } from '@chakra-ui/react'
import axios from 'axios';
import {useState, useEffect} from 'react'
import formatDate from './formatDate';

interface Prediction {
  date: string;
  hour: number;
  predicted_close: number;
};
interface PredictionProps {
  predicted: Prediction[];
}
const FxPredictionResult = () => {
  const [predicted, setPredicted] = useState<Prediction[]>([])

  const fetchPredictions = async() => {
    const url = 'http://127.0.0.1:8000/api/forex/predictions/history/'
    try {
      const res = await axios.get(url)
      let sortedData = res.data
      sortedData = sortedData
        .sort((a: Prediction, b: Prediction) => new Date(b.date).getTime() - new Date(a.date).getTime())
        .slice(0,5)
      setPredicted(sortedData)
      
    }catch(error){
      if (axios.isAxiosError(error)) {
        console.error("fetch prediction failed", error.response?.data);
      } else {
        console.error("fetch prediction failed", error);
      }
    }
  }
  useEffect(()=> {
    fetchPredictions()
  },[])

  // const latestPredictions = predicted
  //   .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
  //   .slice(0, 24);

  return (
    <Box>
        <Table.Root showColumnBorder>
            <Table.Header>
                <Table.Row>
                    <Table.ColumnHeader>Date-Time</Table.ColumnHeader>
                    <Table.ColumnHeader>Prediction</Table.ColumnHeader>
                </Table.Row>
            </Table.Header>
            <Table.Body>
            {
            predicted.length > 0 ? (
            predicted.map((item: Prediction) => (
            <Table.Row key={item.date + '-' + item.hour}>
              <Table.Cell>
                {formatDate(item.date)}
              </Table.Cell>
              <Table.Cell>
                {item.predicted_close}
              </Table.Cell>

            </Table.Row>
            
          ))
        ) : ("no data")
      }
            </Table.Body>
        </Table.Root>
      
    </Box>
  )
}

export default FxPredictionResult