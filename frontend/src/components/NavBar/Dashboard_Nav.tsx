import { Box, HStack } from '@chakra-ui/react'
import { NavLink } from 'react-router'


const Dashboard_Nav = () => {
  return (
    <Box p={"10px"} w={"99vw"} border={"1px solid"} rounded={"7px"} bg={"#d3d3d3"}>
        <HStack w={'100%'} justifyContent={"space-between"} alignItems={"center"} >
            <NavLink to="/home">Home</NavLink>
            <NavLink to='/home/languages'>Learning Languages</NavLink>
            <NavLink to='/home/healthcare'>Examining Health</NavLink>
            <NavLink to='/home/forex'>Foxex</NavLink>
        </HStack>
    </Box>
  )
}

export default Dashboard_Nav