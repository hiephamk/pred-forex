import { Box, HStack, Text, Table } from '@chakra-ui/react'
import axios from 'axios';
import {useState, useEffect} from 'react'
import formatDate from './formatDate';

interface Prediction {
  date: string;
  hour: number;
  open: number;
  high: number;
  low: number;
  close: number;
};
interface PredictionProps {
  predicted: Prediction[];
}

const RealFxData = () => {
  const [realDataFx, setRealDataFx] = useState<Prediction[]>([])

  const fetchPredictions = async() => {
    const url = 'http://localhost:8000/api/forex/real_data/'
    try {
      const res = await axios.get(url)
      let sortedData = res.data
      sortedData = sortedData
        .sort((a: Prediction, b: Prediction) => new Date(b.date).getTime() - new Date(a.date).getTime())
        .slice(0,10)
      setRealDataFx(sortedData)
      
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

  const latestPredictions = realDataFx
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
    .slice(0, 24);

  return (
    <Box>
        <Table.Root showColumnBorder>
            <Table.Header >
                <Table.Row>
                    <Table.ColumnHeader>Date-Time</Table.ColumnHeader>
                    <Table.ColumnHeader>Open</Table.ColumnHeader>
                    <Table.ColumnHeader>High</Table.ColumnHeader>
                    <Table.ColumnHeader>Low</Table.ColumnHeader>
                    <Table.ColumnHeader>Close</Table.ColumnHeader>
                </Table.Row>
            </Table.Header>
            <Table.Body>
            {
            latestPredictions.length > 0 ? (
            latestPredictions.map((item: Prediction) => (
            <Table.Row key={item.date + '-' + item.hour}>
              <Table.Cell>
                {formatDate(item.date)}
              </Table.Cell>
              <Table.Cell>
                {item.open}
              </Table.Cell>
              <Table.Cell>
                {item.high}
              </Table.Cell>
              <Table.Cell>
                {item.low}
              </Table.Cell>
              <Table.Cell>
                {item.close}
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

export default RealFxData